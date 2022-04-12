def tag_and_log_git_status(log_to: str, version: str, exp_name: str, model_name: str) -> None:
    # add git tags for better change tracking
    import subprocess
    gitout = open(log_to, 'a')
    del_tag = f'git tag -d {model_name}_v{version}'
    add_tag = f'git tag -a {model_name}_v{version} -m "{exp_name}"'
    print_branch = "git branch -vv"
    print_status = 'git status'
    print_status2 = 'pip list'
    print_status3 = 'conda list'
    cmds = [del_tag, add_tag, print_branch, print_status, print_status2, print_status3]
    for cmd in cmds:
        o = subprocess.getoutput(cmd)
        gitout.write(f'========={cmd}=========\n{o}\n\n\n')
    gitout.close()
