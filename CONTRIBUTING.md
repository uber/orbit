# Contributing to Orbit

The Orbit project welcome community contributors.
To contribute to it, please follow guidelines here.

The codebase is hosted on Github at https://github.com/uber/orbit.

All code need to follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/) with a few exceptions listed
in [tox.ini](./tox.ini).

Before contributing, please review [outstanding issues](https://github.com/uber/orbit/issues).
If you'd like to contribute to something else, open an issue for discussion first.

# Setup

First fork the repository on `dev` branch

```bash
$ git clone --single-branch --branch dev https://github.com/uber/orbit.git
```

# Making Changes

Create a new branch to develop new code

```bash
$ cd orbit
$ git checkout feat-my-new-feature # our branch naming convention
$ pip install -r requirements.txt # install dependencies
$ pip install -e . # install with dev mode
```

# Test

## Prerequisites

Install orbit required dependencies for test.

```bash
$ pip install -r requirements-test.txt
```

## Testing

After your changes and before submitting a pull request, make sure the change to pass all tests and test coverage
to be at least 70%.

```bash
$ pytest -vs tests/ --cov orbit/
```

## Linting

You can run black linting to lint the code style.

### Linting one single file

```bash
$ black <file path>
```

### Linting every file under the current directory

```bash
$ black .
```

### Outputting the code change black would have done without actually making change

```bash
$ black --diff <file path>
```

# Submission

In your PR, please include:

- Changes made
- Links to related issues/PRs
- Tests
- Dependencies
- References

Please add the core Orbit contributors as reviewers.

## Merging and Releasing versions

We use squash and merge for changes onto `dev` branch. However, due to history comparison, from `dev` to `release`
and `master`, we use rebase and merge. For release details, please refer to [RELEASE.md](./RELEASE.md)
