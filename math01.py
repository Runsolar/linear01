

w = [1, 2, 3, 4]

v = [1, 2, 3, 4]

k = 4


#dot = sum(list(map(lambda x, y: x*y, w, v)))

dot = list(map(lambda x, y: x*y, w, v if type(k) is list else [k for _ in range(len(w))]))

if type(k) is list:
    print(sum(dot))
else:
    print(dot)



