# microgpt-c

A C implementation of Karpathy's microGPT - a minimal, dependency-free GPT implementation with autograd.

## Overview

This project is a faithful C port of [Andrej Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), implementing a complete GPT transformer with:

- **Automatic differentiation (autograd)** - Build computation graphs and backpropagate gradients
- **Transformer architecture** - Multi-head attention, MLP blocks, residual connections, RMSNorm
- **Adam optimizer** - With bias correction and learning rate decay
- **Text generation** - Temperature-controlled sampling

The implementation follows the original Python algorithm closely, demonstrating how the core concepts of modern LLMs can be expressed in pure C with no external dependencies beyond the standard library.

## Architecture

- **Layers**: 1 transformer layer
- **Embedding dimension**: 16
- **Attention heads**: 4 (head dimension: 4)
- **Context window**: 16 tokens
- **Vocabulary**: Byte-level (up to 256 characters)

## Project Structure

```
microgpt.c      - Main implementation (autograd, forward pass, training, inference)
microgpt.h      - Header file with data structures and function declarations
microgpt.py     - Original Python reference implementation by Karpathy
Makefile        - Build configuration for GCC
input.txt       - Training data (one sample per line)
```

## Building

Requires GCC with C99 support. The project uses a large stack size (32MB) to accommodate the deep autograd computation graphs.

```bash
# Build the project
make

# Clean build artifacts
make clean
```

The compiled binary will be `microgpt` (Linux/macOS) or `microgpt.exe` (Windows).

## Usage

```bash
# Train on default dataset (input.txt)
./microgpt

# Train on custom dataset
./microgpt my_dataset.txt
```

### Training

The model trains for 1000 steps with the following hyperparameters:
- Learning rate: 0.01 (with linear decay)
- Adam beta1: 0.85
- Adam beta2: 0.99
- Batch size: 1 document per step

### Inference

After training, the model generates 20 samples using temperature-controlled sampling (temperature=0.5).

## Features

### Autograd System

The implementation includes a complete automatic differentiation engine:
- **Computation graph**: Nodes track forward pass values and build a graph of operations
- **Backpropagation**: Iterative topological sort using heap-allocated stack (avoids C stack overflow)
- **Gradient accumulation**: Local gradients are cached during forward pass for efficient backward

### Memory Management

Careful memory management is crucial in C:
- Computation graphs are freed after each training step using BFS traversal
- Parameter gradients are zeroed after each optimizer step
- KV cache pointers are cleared after graph cleanup

### Performance Considerations

- Uses iterative (not recursive) topological sort to prevent stack overflow on large graphs
- 32MB stack allocation configured via linker flags
- Computation graphs can reach ~350K nodes for typical forward passes

## Comparison with Python Original

| Feature | Python | C |
|---------|--------|---|
| Dependencies | None (pure Python) | None (pure C, stdlib only) |
| Autograd | Recursive `build_topo` | Iterative with heap stack |
| Memory | GC-managed | Manual (malloc/free) |
| Speed | ~minutes | ~seconds |
| Lines of code | ~200 | ~720 |

## Dataset Format

The training data should be a text file with one sample per line. The default `input.txt` contains common English names (e.g., emma, olivia, ava, isabella, ...).

The tokenizer operates at the character level:
- Unique characters in the dataset become tokens (IDs 0 to N-1)
- A special BOS (Beginning of Sequence) token is added with ID N

## Model Architecture Details

Follows GPT-2 design with minor modifications:
- **RMSNorm** instead of LayerNorm
- **No biases** in linear layers
- **ReLU** activation instead of GeLU
- **Single transformer layer** for simplicity

### Forward Pass Flow

```
token_id, pos_id
  → Token + Position Embedding
  → RMSNorm
  → Multi-Head Attention (with residual)
    → Q, K, V linear projections
    → Scaled dot-product attention
    → Output projection
  → RMSNorm
  → MLP (with residual)
    → Linear → ReLU → Linear
  → LM Head (vocab_size logits)
  → Softmax → Cross-entropy loss
```

## Extending the Model

To modify the architecture, edit the constants in `microgpt.h`:

```c
#define N_LAYER     1   // Number of transformer layers
#define N_EMBD      16  // Embedding dimension
#define BLOCK_SIZE  16  // Context window size
#define N_HEAD      4   // Number of attention heads
#define NUM_STEPS   1000 // Training steps
```

## Limitations

This is a minimal educational implementation:
- Small model capacity (2K-3K parameters)
- Single-batch training (no mini-batch support)
- Character-level tokenizer (not subword)
- No validation or early stopping
- Computation graphs rebuilt from scratch each step

## License

This project is a port of Karpathy's microgpt. The original code is available at:
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## Acknowledgments

- **Andrej Karpathy** - Original microgpt implementation and educational content
- This C port maintains the spirit of the original: minimal, educational, dependency-free

## References

- [Original microgpt (Python)](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
