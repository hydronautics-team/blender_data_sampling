import yaml

with open('params.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)

print()
print(data['render_settings']['NUM_IMAGES'])
