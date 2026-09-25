from neuroez_c.task2.native_ssl_n2.split import split
def test_split_isolated():
 meta={str(i):{'center':'a','outcome_success':i%2} for i in range(8)}; train,val=split(list(meta),meta,.2,42); assert not set(train)&set(val)
