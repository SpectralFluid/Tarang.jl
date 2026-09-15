# Contributing

Fork, clone, and submit PRs to `main`. Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.

Start with [Architecture and Codebase Structure](architecture.md) for file
ownership and the solver runtime path, and [Testing](testing.md) for the CPU,
GPU, and MPI runners.

## Updating the web documentation

Edit published pages under `docs/src/` and add new navigation entries to
`docs/make.jl`. Files elsewhere under `docs/`, including design plans, are not
part of the web manual unless explicitly included.

Build against this checkout from the repository root:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html` to inspect the result. The Documentation workflow
builds pull requests and publishes previews for branches in this repository.
Pushes to `main` publish the **dev** manual at
[spectralfluid.github.io/Tarang.jl/dev/](https://spectralfluid.github.io/Tarang.jl/dev/);
release tags update the versioned manual and **stable** alias. A PR preview
does not update the `main` manual.

**Style**: 4-space indent, 92-char lines, `snake_case` functions, `CamelCase` types.

**Help**: [Issues](https://github.com/SpectralFluid/Tarang.jl/issues) | [Discussions](https://github.com/SpectralFluid/Tarang.jl/discussions)
