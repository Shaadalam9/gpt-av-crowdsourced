import os

main_module_dir = os.path.dirname(__file__)
root_dir = os.path.normpath(os.path.join(main_module_dir, '..'))
cache_dir = os.path.join(root_dir, '_cache')
log_dir = os.path.join(root_dir, '_logs')
output_dir = os.path.join(root_dir, '_output')

for path in [output_dir, log_dir]:
    if not os.path.exists(path):
        os.makedirs(path)
