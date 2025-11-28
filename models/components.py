import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))


class TopKRouter(nn.Module):
    """Router that selects top-k experts for each token"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1  # Standard deviation for noise during training

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs



class CredalRouter(nn.Module):
    """
    Credal Router that dynamically selects top-k experts based on epistemic uncertainty.
    Based on 'Credal Transformer' and Evidential Deep Learning.
    """
    def __init__(self, d_model: int, num_experts: int, base_top_k: int = 1, credal_lambda: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.base_top_k = base_top_k
        self.credal_lambda = credal_lambda
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1
        # Track expert selection statistics
        self.expert_counts = []
        self.uncertainties = []
        self.selected_experts = []  # Track which experts were selected

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - router_weights: [batch, seq, max_k] (padded with zeros if k < max_k)
            - expert_indices: [batch, seq, max_k]
            - router_probs: [batch, seq, num_experts]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Compute logits
        logits = self.gate(x)
        
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
            
        # 2. Compute Evidence (softplus for stability)
        evidence = F.softplus(logits)
        
        # 3. Compute Dirichlet parameters
        alpha = evidence + 1.0
        total_evidence = torch.sum(alpha, dim=-1, keepdim=True)
        
        # 4. Compute Epistemic Uncertainty (Vacuity)
        # Normalized uncertainty: U = N / (N + S)
        # This gives values in [0, 1] where:
        # - High uncertainty (→1): When total_evidence is small (uncertain predictions)
        # - Low uncertainty (→0): When total_evidence is large (confident predictions)
        uncertainty = self.num_experts / (self.num_experts + total_evidence)  # [batch, seq, 1]
        
        # 5. Determine dynamic k per token
        # k = base_k + floor(lambda * U)
        # We need a fixed k for the batch to keep tensor shapes consistent in this implementation,
        # so we take the mean uncertainty of the batch or max.
        # For true per-token dynamic k, we'd need sparse tensors or masking.
        # Here we implement a "Batch-Adaptive Top-K" for efficiency.
        avg_uncertainty = uncertainty.mean()
        dynamic_k = self.base_top_k + int(self.credal_lambda * avg_uncertainty.item())
        dynamic_k = min(max(dynamic_k, 1), self.num_experts)
        
        # 6. Standard routing with the calculated k
        router_probs = F.softmax(logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(logits, dynamic_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Track statistics (only during training to avoid memory issues)
        if self.training:
            self.expert_counts.append(dynamic_k)
            self.uncertainties.append(avg_uncertainty.item())
            # Store which experts were selected (sample from batch for efficiency)
            self.selected_experts.append(top_k_indices[0, 0, :].cpu().tolist())  # First token of first batch
        
        return top_k_weights, top_k_indices, router_probs


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        use_credal_routing: bool = False,
        credal_lambda: float = 1.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight
        self.use_credal_routing = use_credal_routing

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        if use_credal_routing:
            # For Credal, top_k acts as the base_top_k (minimum experts)
            self.router = CredalRouter(d_model, num_experts, base_top_k=1, credal_lambda=credal_lambda)
        else:
            self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight
