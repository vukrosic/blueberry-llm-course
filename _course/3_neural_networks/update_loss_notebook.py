import json

# Load the existing notebook
notebook_path = '400_loss_functions.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# New cells to add (before the Summary section which is at index -1)
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Deep Dive: Cross Entropy Step-by-Step\n",
            "\n",
            "Let's break down exactly what happens inside Cross Entropy Loss!"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 1: Softmax - Converting Logits to Probabilities\n",
            "\n",
            "Neural networks output raw scores called **logits**. We use **Softmax** to convert them to probabilities:\n",
            "\n",
            "$\\text{Softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j} e^{x_j}}$\n",
            "\n",
            "This ensures all probabilities sum to 1.0!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Single sample: predicting Cat, Dog, or Bird\n",
            "logits = torch.tensor([2.0, 1.0, 0.1])\n",
            "\n",
            "# Manual Softmax calculation\n",
            "exp_logits = torch.exp(logits)\n",
            "probabilities = exp_logits / exp_logits.sum()\n",
            "\n",
            "print(\"Logits:\", logits)\n",
            "print(\"\\nExponentials:\")\n",
            "print(f\"  e^2.0 = {exp_logits[0]:.3f}\")\n",
            "print(f\"  e^1.0 = {exp_logits[1]:.3f}\")\n",
            "print(f\"  e^0.1 = {exp_logits[2]:.3f}\")\n",
            "print(f\"  Sum = {exp_logits.sum():.3f}\")\n",
            "\n",
            "print(\"\\nProbabilities after Softmax:\")\n",
            "print(f\"  P(Cat)  = {probabilities[0]:.3f} ({probabilities[0]*100:.1f}%)\")\n",
            "print(f\"  P(Dog)  = {probabilities[1]:.3f} ({probabilities[1]*100:.1f}%)\")\n",
            "print(f\"  P(Bird) = {probabilities[2]:.3f} ({probabilities[2]*100:.1f}%)\")\n",
            "print(f\"\\nSum of probabilities: {probabilities.sum():.6f}\")\n",
            "\n",
            "# Verify with PyTorch's Softmax\n",
            "softmax = nn.Softmax(dim=0)\n",
            "probs_torch = softmax(logits)\n",
            "print(f\"\\nPyTorch Softmax: {probs_torch}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 2: Cross Entropy - Correct Prediction\n",
            "\n",
            "When the model predicts correctly with **high confidence**, loss is **low**.\n",
            "\n",
            "$\\text{CE} = -\\sum y_{\\text{true}} \\cdot \\log(p_{\\text{pred}})$\n",
            "\n",
            "For a single correct class, this simplifies to: $\\text{CE} = -\\log(p_{\\text{correct class}})$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Scenario: Image is a Cat (class 0), model predicts Cat strongly\n",
            "logits = torch.tensor([[2.0, 1.0, 0.1]])  # Shape: (1, 3)\n",
            "target = torch.tensor([0])  # True class: Cat\n",
            "\n",
            "# Step 1: Get probabilities\n",
            "probs = torch.softmax(logits, dim=1)\n",
            "print(\"Predicted probabilities:\")\n",
            "print(f\"  Cat:  {probs[0][0]:.4f} (65.9%)\")\n",
            "print(f\"  Dog:  {probs[0][1]:.4f} (24.2%)\")\n",
            "print(f\"  Bird: {probs[0][2]:.4f} (9.9%)\")\n",
            "\n",
            "# Step 2: Manually calculate Cross Entropy\n",
            "# Since true class is 0 (Cat), we only care about P(Cat)\n",
            "prob_correct_class = probs[0][target[0]]\n",
            "manual_ce = -torch.log(prob_correct_class)\n",
            "\n",
            "print(f\"\\nProbability of correct class (Cat): {prob_correct_class:.4f}\")\n",
            "print(f\"Manual Cross Entropy: -log({prob_correct_class:.4f}) = {manual_ce:.4f}\")\n",
            "\n",
            "# Step 3: Verify with PyTorch\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "loss = criterion(logits, target)\n",
            "print(f\"PyTorch Cross Entropy: {loss.item():.4f}\")\n",
            "\n",
            "print(\"\\n✅ Loss is LOW because prediction was CORRECT and CONFIDENT!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 3: Cross Entropy - Wrong Prediction\n",
            "\n",
            "When the model predicts **incorrectly** with high confidence, loss is **very high**!\n",
            "\n",
            "This heavily penalizes confident mistakes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Scenario: Image is a Bird (class 2), but model predicts Cat very strongly!\n",
            "logits = torch.tensor([[4.0, 0.0, 0.0]])  # Model is 95%+ sure it's a Cat\n",
            "target = torch.tensor([2])  # True class: Bird\n",
            "\n",
            "# Step 1: Get probabilities\n",
            "probs = torch.softmax(logits, dim=1)\n",
            "print(\"Predicted probabilities:\")\n",
            "print(f\"  Cat:  {probs[0][0]:.4f} ({probs[0][0]*100:.1f}%)\")\n",
            "print(f\"  Dog:  {probs[0][1]:.4f} ({probs[0][1]*100:.1f}%)\")\n",
            "print(f\"  Bird: {probs[0][2]:.4f} ({probs[0][2]*100:.1f}%) ← TRUE CLASS\")\n",
            "\n",
            "# Step 2: Calculate Cross Entropy manually\n",
            "prob_correct_class = probs[0][target[0]]\n",
            "manual_ce = -torch.log(prob_correct_class)\n",
            "\n",
            "print(f\"\\nProbability of correct class (Bird): {prob_correct_class:.4f}\")\n",
            "print(f\"Manual Cross Entropy: -log({prob_correct_class:.4f}) = {manual_ce:.4f}\")\n",
            "\n",
            "# Verify with PyTorch\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "loss = criterion(logits, target)\n",
            "print(f\"PyTorch Cross Entropy: {loss.item():.4f}\")\n",
            "\n",
            "print(\"\\n❌ Loss is VERY HIGH because model was WRONG and CONFIDENT!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 4: Understanding the Math\n",
            "\n",
            "Why does $-\\log(p)$ work so well?\n",
            "\n",
            "Let's visualize how loss changes with prediction confidence:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Range of probabilities from near 0 to 1\n",
            "probabilities = torch.linspace(0.01, 0.99, 100)\n",
            "losses = -torch.log(probabilities)\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.plot(probabilities.numpy(), losses.numpy(), linewidth=2)\n",
            "plt.xlabel('Predicted Probability of Correct Class', fontsize=12)\n",
            "plt.ylabel('Cross Entropy Loss', fontsize=12)\n",
            "plt.title('How Cross Entropy Penalizes Predictions', fontsize=14, fontweight='bold')\n",
            "plt.grid(True, alpha=0.3)\n",
            "\n",
            "# Add annotations\n",
            "plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Random guess (50%)')\n",
            "plt.axhline(y=0, color='green', linestyle='--', alpha=0.7)\n",
            "\n",
            "# Annotate key points\n",
            "plt.text(0.95, 0.5, 'High confidence\\nCORRECT\\n(Low loss)', \n",
            "         fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))\n",
            "plt.text(0.05, 4.5, 'High confidence\\nWRONG\\n(High loss)', \n",
            "         fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))\n",
            "\n",
            "plt.legend()\n",
            "plt.ylim(0, 5)\n",
            "plt.show()\n",
            "\n",
            "print(\"Key insights:\")\n",
            "print(\"  • P = 1.0 → Loss ≈ 0 (perfect prediction)\")\n",
            "print(\"  • P = 0.5 → Loss ≈ 0.69 (random guess)\")\n",
            "print(\"  • P → 0 → Loss → ∞ (confident but wrong!)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 5: Batch Cross Entropy\n",
            "\n",
            "In practice, we compute loss over multiple samples and take the mean."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Batch of 3 images\n",
            "logits = torch.tensor([\n",
            "    [2.0, 1.0, 0.1],   # Sample 1: Predicts Cat\n",
            "    [0.5, 2.5, 0.3],   # Sample 2: Predicts Dog  \n",
            "    [4.0, 0.0, 0.0]    # Sample 3: Predicts Cat\n",
            "])\n",
            "targets = torch.tensor([0, 1, 2])  # True: Cat, Dog, Bird\n",
            "\n",
            "# Calculate loss for each sample individually\n",
            "probs = torch.softmax(logits, dim=1)\n",
            "\n",
            "print(\"Individual sample analysis:\\n\")\n",
            "for i in range(3):\n",
            "    true_class = targets[i].item()\n",
            "    prob_true = probs[i][true_class]\n",
            "    loss_i = -torch.log(prob_true)\n",
            "    \n",
            "    predicted_class = logits[i].argmax().item()\n",
            "    correct = \"✅\" if predicted_class == true_class else \"❌\"\n",
            "    \n",
            "    print(f\"Sample {i+1}: {correct}\")\n",
            "    print(f\"  True class: {true_class}, Predicted: {predicted_class}\")\n",
            "    print(f\"  P(true class) = {prob_true:.4f}\")\n",
            "    print(f\"  Loss = {loss_i:.4f}\\n\")\n",
            "\n",
            "# PyTorch computes mean of all losses\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "total_loss = criterion(logits, targets)\n",
            "\n",
            "# Manual mean\n",
            "manual_losses = torch.tensor([loss_i for i in range(3)])\n",
            "for i in range(3):\n",
            "    manual_losses[i] = -torch.log(probs[i][targets[i]])\n",
            "    \n",
            "mean_loss = manual_losses.mean()\n",
            "\n",
            "print(f\"Average loss (manual): {mean_loss:.4f}\")\n",
            "print(f\"PyTorch CrossEntropyLoss: {total_loss:.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 6: Why Not Just Use Accuracy?\n",
            "\n",
            "Cross Entropy is **differentiable** and provides **gradient information** for learning.\n",
            "\n",
            "Accuracy only tells you if you're right or wrong, but not **how confident** you should be."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Two scenarios: both predict correctly, but with different confidence\n",
            "\n",
            "# Scenario A: Hesitant but correct\n",
            "logits_a = torch.tensor([[1.1, 0.9, 0.8]])  # Barely predicts class 0\n",
            "target = torch.tensor([0])\n",
            "\n",
            "probs_a = torch.softmax(logits_a, dim=1)\n",
            "loss_a = nn.CrossEntropyLoss()(logits_a, target)\n",
            "\n",
            "# Scenario B: Confident and correct\n",
            "logits_b = torch.tensor([[3.0, 0.0, 0.0]])  # Very confident about class 0\n",
            "probs_b = torch.softmax(logits_b, dim=1)\n",
            "loss_b = nn.CrossEntropyLoss()(logits_b, target)\n",
            "\n",
            "print(\"Both scenarios predict class 0 correctly:\\n\")\n",
            "\n",
            "print(\"Scenario A (Hesitant):\")\n",
            "print(f\"  Probabilities: {probs_a[0]}\")\n",
            "print(f\"  P(class 0) = {probs_a[0][0]:.4f}\")\n",
            "print(f\"  Cross Entropy Loss: {loss_a:.4f}\\n\")\n",
            "\n",
            "print(\"Scenario B (Confident):\")\n",
            "print(f\"  Probabilities: {probs_b[0]}\")\n",
            "print(f\"  P(class 0) = {probs_b[0][0]:.4f}\")\n",
            "print(f\"  Cross Entropy Loss: {loss_b:.4f}\\n\")\n",
            "\n",
            "print(\"📊 Analysis:\")\n",
            "print(f\"  Accuracy: Both are 100% correct\")\n",
            "print(f\"  Cross Entropy: Rewards confidence! ({loss_b:.4f} < {loss_a:.4f})\")"
        ]
    }
]

# Update the Summary section number from 4 to 5
nb['cells'][-1]['source'][0] = "## 5. Summary\n"

# Insert new cells before the last cell (Summary)
nb['cells'] = nb['cells'][:-1] + new_cells + [nb['cells'][-1]]

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)

print(f"✅ Successfully added {len(new_cells)} new cells to the notebook!")
print("The deep dive section on Cross Entropy has been added before the Summary.")
