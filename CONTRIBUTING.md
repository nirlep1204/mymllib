# Contributing to mymllib

Hey! Thanks for being interested in contributing. This project is a personal learning exercise, but if you spot a bug, want to optimize an algorithm, or have ideas for new features — I'd genuinely appreciate the help.

## How to contribute

1. **Fork** this repo and clone it locally.
2. Create a new branch: `git checkout -b my-fix`
3. Make your changes.
4. Run the tests to make sure nothing breaks:
   ```bash
   pytest tests/
   ```
5. Open a Pull Request with a short description of what you changed and why.

## What I'd love help with

- Bug fixes (especially edge cases in algorithms like SVM or GMM)
- Performance improvements (vectorization, reducing redundant loops)
- New algorithms (see the Roadmap in README.md)
- Better documentation or examples
- Typo fixes (yes, even those matter!)

## Style

- Keep it simple. This library was built to be readable, not clever.
- Use clear variable names. `weights` not `w`. `learning_rate` not `lr`.
- Add comments where the math isn't obvious.

## Be kind

This started as a solo learning project. If you see something that could be done better, that's great — just be constructive about it. We're all here to learn.
