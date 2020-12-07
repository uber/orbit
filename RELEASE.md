# Orbit Release Process

## Full Release
1. Create a release branch from `dev`
    - e.g. `release/v1.0.15`
2. Update the version number in `orbit/__init__.py`. This version number will propagate to `docs/conf.py`, `setup.cfg`, and `setup.py`.
3. Commit changes
4. If necessary, additional PRs may be merged to the release branch directly, but this should be reserved for bug fixes only and should not add or change any features
5. Merge the release branch to both `dev` and `master`
6. Draft a new release: https://github.com/uber/orbit/releases/new
    - Select the master branch as the target branch
    - Use version number for both the tag and title
    - Add a bulleted list of changes in the description
7. Click `Publish Release` once all changes are finalized and description is updated


## Quick Release
Sometimes we just want to release a patch, and no subsequent commits are needed on the release branch.
In this case, we can avoid creating the branch and create a release directly from dev.

1. From `dev`, update the version number in `orbit/__init__.py`.
2. Commit changes
3. Merge to `master`
4. Draft a new release: https://github.com/uber/orbit/releases/new
    - Select the master branch as the target branch
    - Use version number for both the tag and title
    - Add a bulleted list of changes in the description
    
    
## Hotfix
Sometimes we may need to address a bug fix directly from master after a release, but `dev` may have moved on with new commits.

1. Create a hotfix branch from master and update the version number
2. Make fix
3. Merge changes into `master`
4. Draft a new release: https://github.com/uber/orbit/releases/new
    - Select the master branch as the target branch
    - Use version number for both the tag and title
    - Add a bulleted list of changes in the description
5. Merge changes into `dev`
