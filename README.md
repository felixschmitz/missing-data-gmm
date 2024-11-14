# missing-data-GMM

Reproducing the method and results of the paper "A GMM approach for dealing with missing
data on regressors" by Jason Abrevaya and Stephen G. Donald (2017). The paper and
replication code can be found
[here](https://direct.mit.edu/rest/article/99/4/657/58390/A-GMM-Approach-for-Dealing-with-Missing-Data-on).

## Table of Contents

1. 🚀 [Getting Started](#getting-started)
1. 📦 [Pixi](#1-pixi)
1. ✅ [Pre-commit Hooks](#2-pre-commit-hooks)
1. 🛠️ [Using Pytask](#3-using-pytask)
1. 🌳 [Git Workflow](#4-git-workflow)
1. 📋 [Important Commands Summary](#5-important-commands-summary)

## Getting Started

This guide will walk you through setting up a shared environment to ensure that all
collaborators use the same package versions and adhere to code quality standards. The
setup involves using **Pixi** for environment management, **pre-commit hooks** for code
quality, and **Pytask** for task management.

### 1. Pixi

Pixi is a tool that helps manage project dependencies (packages) and environments,
ensuring that the code runs consistently on different machines. Once installed, Pixi
will automatically handle the correct versions of packages so that we don’t encounter
"it works on my machine" issues.

#### Downloading and Installing Pixi

To install Pixi, follow the instructions on the
[Pixi GitHub page](https://github.com/prefix-dev/pixi?tab=readme-ov-file#installation).
Once downloaded, open your terminal, navigate to the project directory, and run:

```sh
pixi install
```

#### Activating the Pixi Shell

The Pixi shell is an isolated environment for running commands specific to this project.
To activate it, use:

```sh
pixi shell
```

You should always activate the Pixi shell before starting work on the project. This
ensures that all commands run in the shared environment with the correct packages.

### 2. Pre-commit Hooks

Pre-commit hooks are automated checks that help maintain code quality and consistency.
They run before each commit to catch issues such as formatting errors, style violations,
or other predefined standards.

#### Installing Pre-commit Hooks

After activating the Pixi shell, install the pre-commit hooks by running:

```sh
pre-commit install
```

This only needs to be done once. Afterward, each time you commit code, the hooks will
automatically check it against our quality standards.

### 3. Using Pytask

Pytask is our tool for managing tasks within the project. It helps automate workflows,
such as running scripts or executing specific parts of the code in an organized way.

#### Collecting All Pytasks

To identify all available tasks in the project, run:

```sh
pytask collect
```

This command will show you the tasks available to execute within this environment.

#### Running Tasks

To run all tasks, use:

```sh
pytask
```

If you want to run a specific task, refer to the
[Pytask documentation](https://pytask-dev.readthedocs.io/en/latest/tutorials/selecting_tasks.html)
for more details.

#### Project Structure

Here's an overview of the project's structure to help you get oriented:

- **`src/`**: Contains the main codebase for the project.
- **`data/`**: Contains data files used within the project.

This layout ensures that code and data are separated for clarity and organization.

#### Creating a New Task

To add a new task to automate a specific action (e.g., data preprocessing, model
training), you can use Pytask. For more guidance, refer to the
[Pytask documentation](https://pytask-dev.readthedocs.io/en/stable/).

### 4. Git Workflow

This project uses a branch-based approach to keep the `main` branch stable and ensure
code quality. Each feature or change should be developed in a separate branch, reviewed,
and then merged into `main`. Here’s a step-by-step guide:

#### Creating a New Feature Branch

1. **Start from the `main` branch**. Make sure your local `main` is up-to-date:

   ```sh
   git checkout main
   git pull origin main
   ```

1. **Create a new branch** for your feature or change. Use descriptive names for your
   branch, such as `your-feature-name`:

   ```sh
   git checkout -b your-feature-name
   ```

#### Committing Changes

1. **Make changes** to the code as needed in your feature branch.

1. **Add and commit** your changes. Remember to write clear, descriptive commit
   messages:

   ```sh
   git add .
   git commit -m "A clear, descriptive message about your changes"
   ```

   *Note:* If you haven’t installed the pre-commit hooks, follow the setup in the README
   to ensure all commits meet code quality standards.

#### Pushing Changes

1. **Push your feature branch** to the remote repository:
   ```sh
   git push origin your-feature-name
   ```

#### Creating a Pull Request (PR)

1. Go to the GitHub repository and open a **Pull Request** from your feature branch to
   `main`.
1. Request a review if necessary. This ensures another team member checks the changes
   before merging.

#### Merging the Feature Branch into `main`

1. After approval, **merge** your feature branch into `main`. You have permissions to
   use the GitHub UI to merge the PR.
1. **Delete** the feature branch after merging to keep the repository organized.

______________________________________________________________________

### 5. Important Commands Summary

Here’s a quick reference for the commands we’ve covered:

1. **Activate Pixi Shell**:

   ```sh
   pixi shell
   ```

1. **Install Pre-commit Hooks** (only once):

   ```sh
   pre-commit install
   ```

1. **Collect Tasks**:

   ```sh
   pytask collect
   ```

1. **Run All Tasks**:

   ```sh
   pytask
   ```

1. **Git Commands**:

   - **Update `main` and create feature branch**:

     ```sh
     git checkout main
     git pull origin main
     git checkout -b your-feature-name
     ```

   - **Add and commit changes**:

     ```sh
     git add .
     git commit -m "Your commit message"
     ```

   - **Push feature branch**:

     ```sh
     git push origin your-feature-name
     ```

These commands should cover most routine actions you’ll need when working on the
project. This workflow ensures that each change is tracked, reviewed, and integrated
systematically, keeping `main` stable and reliable.

For further details on the workflow and tools, refer to this
[useful page](https://effective-programming-practices.vercel.app/landing-page.html).
