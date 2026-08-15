# ALW Systems – GPU-Accelerated Signal Processing Stack

**97% custom CUDA code.**  
Double-double arithmetic (1e-32 precision). Adaptive noise estimation (AMAD-X). GPU linear solver (AEDS). Sparse decomposition (EOP). Dictionary learning (EOD). Frequency control (AFC).

## 🚀 Why This Stack
- **AEDS:** 15-20x speedup vs CPU on 1000x1000 matrices
- **AMAD-X:** 8-10x speedup, robust multi-scale noise estimation
- **EOP/EOD:** Full sparse decomposition & dictionary training on GPU
- **Production-ready:** Custom memory pools, LRU cache, RAII guards

## 📁 Structure
- `src/` — CUDA kernels & core logic
- `docs/` — Architecture & benchmarks (will be soon)
- `examples/` — Ready-to-run demos

## 🔧 Quick Build & Run

```bash
# Clone the repository
git clone https://github.com/vallais-dev/alw-systems-portfolio
cd alw-systems-portfolio

# Compile the demo
nvcc -O3 -arch=sm_61 -I. examples/simple_demo.cu -o simple_demo -lcuda -lcudart -lm

# Run
./simple_demo

```

## 🔒 License
**CC BY-NC 4.0** — Free for review & academic use. Commercial use requires separate license.

## 📬 Contact (Available for hire)
Email: vallais.one@gmail.com  
Telegram: @vallais_one
X: @VallaisOne

---
*Built with ❤️ and CUDA. 97% custom code. 100% performance.*
