from tqdm.auto import tqdm
def bar(items,desc,leave=True):return tqdm(items,desc=desc,leave=leave,dynamic_ncols=True)
