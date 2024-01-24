import subprocess


def execute_commands_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        command = file.read()
        # if len(lines) > 1:
        #     print("Note: there are %s lines of commands." % len(lines))
    command = command.replace('\n', ' ')
    subprocess.run(command, shell=True)


if __name__ == '__main__':
    command_file = 'commands.txt'
    execute_commands_from_file(command_file)
