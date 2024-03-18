from pathlib import Path

def tag_and_log_git_status(log_to: str, version: str, exp_name: str, model_name: str) -> None:
    # add git tags for better change tracking
    import subprocess
    gitout = open(log_to, 'a', encoding='utf-8')
    del_tag = f'git tag -d {model_name}_v{version}'
    add_tag = f'git tag -a {model_name}_v{version} -m "{exp_name}"'
    print_branch = "git branch -vv"
    print_status = 'git status'
    print_status2 = f'pip freeze > {str(Path(log_to).expanduser().parent)}/requirements_pip.txt'
    print_status3 = f'conda list -e > {str(Path(log_to).expanduser().parent)}/requirements_conda.txt'
    cmds = [del_tag, add_tag, print_branch, print_status, print_status2, print_status3]
    for cmd in cmds:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, encoding="utf-8", universal_newlines=True, shell=True)
        o, err = p.communicate()
        gitout.write(f'========={cmd}=========\n{o}\n\n\n')
    gitout.close()
