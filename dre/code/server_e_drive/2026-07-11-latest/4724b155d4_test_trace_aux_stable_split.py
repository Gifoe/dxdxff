from neuroez_c.task2.trace_aux_stable.split import make_split
def test_split_disjoint():
 m={str(i):{'center':'a' if i<4 else 'b','outcome_success':i%2} for i in range(10)};a,b=make_split(list(m),m,.2,42);assert not set(a)&set(b)
