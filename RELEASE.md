# Orbit Release Process

## Full Release
1. Create a **doc-refresh** branch. Add the logs of changes under 
   [changelog](https://github.com/uber/orbit/blob/dev/docs/changelog.rst).  
   The logs can be reused on later draft of release under Github.
   Rerun notebooks under **docs/tutorails** for defined readthedocs triggered job.
2. Submit a PR for dev for the changes of any documentation. 
3. After the approval and merge from previous PR, create a release branch from `dev`
    - e.g. `release-v1.0.15`
4. Update the version number in `orbit/__init__.py`. This version number will propagate to `docs/conf.py`, `setup.cfg`, and `setup.py`.
5. Commit changes
6. Test PyPI deployment locally by `python3 -m twine check dist`   
7. If necessary, additional PRs may be merged to the release branch directly, but this should be reserved for bug fixes only and should not add or change any features
8. Merge the release branch to both `dev` and `master`
9. Draft a new release: https://github.com/uber/orbit/releases/new
    - Select the `master` as the target branch
    - Use version number for both the tag and title e.g. `v1.0.15`
    - Add a bulleted list of changes in the description; this can be similar to change logs from step 1.
10. Click `Publish Release` once all changes are finalized and description is updated
11. All the documentation should be refreshed and can be found in https://orbit-ml.readthedocs.io/en/stable/


## Quick Release
Sometimes we just want to release a patch, and no subsequent commits are needed on the release branch.
In this case, we can avoid creating the branch and create a release directly from dev.

1. From `dev`, update the version number in `orbit/__init__.py`.
2. Commit changes
3. Merge to `master`
4. Draft a new release: https://github.com/uber/orbit/releases/new
    - Select the master branch as the target branch
    - Use version number for both the tag and title
    - Add a bulleted list of changes in the description as well as
      [changelog](https://github.com/uber/orbit/blob/dev/docs/changelog.rst).
    
    
## Hotfix
Sometimes we may need to address a bug fix directly from master after a release, but `dev` may have moved on with new commits.

1. Create a hotfix branch from master and update the version number
2. Make fix
3. Merge changes into `master`
4. Draft a new release: https://github.com/uber/orbit/releases/new
    - Select the master branch as the target branch
    - Use version number for both the tag and title
    - Add a bulleted list of changes in the description as well as
    [changelog](https://github.com/uber/orbit/blob/dev/docs/changelog.rst).
5. Merge changes into `dev`
