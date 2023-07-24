import os.path
import os

sub_fold = 'measuring results'
filename = 'example.log'
cwd = os.path.abspath(os.getcwd())
full_path = os.path.join(cwd,sub_fold, filename)
print(full_path)