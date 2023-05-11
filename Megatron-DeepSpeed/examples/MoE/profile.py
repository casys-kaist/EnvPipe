import math
import pprint

total_raw = []
raw = []

f = open("0.out", "r")
log = False
one_iter = []
while True:
    line = f.readline()
    if not line: break
    if "_" not in line and not log:
        log = True
        if len(one_iter) != 0:
            raw.append(one_iter)
        one_iter = []
    elif "_" in line and log:
        log = False

    if log:
        one_iter.append(float(line.strip()))
f.close()
total_raw.append(raw)

raw = []
f = open("1.out", "r")
log = False
one_iter = []
while True:
    line = f.readline()
    if not line: break
    if "_" not in line and not log:
        log = True
        if len(one_iter) != 0:
            raw.append(one_iter)
        one_iter = []
    elif "_" in line and log:
        log = False

    if log:
        one_iter.append(float(line.strip()))
f.close()
total_raw.append(raw)

raw = []
f = open("2.out", "r")
log = False
one_iter = []
while True:
    line = f.readline()
    if not line: break
    if "_" not in line and not log:
        log = True
        if len(one_iter) != 0:
            raw.append(one_iter)
        one_iter = []
    elif "_" in line and log:
        log = False

    if log:
        one_iter.append(float(line.strip()))
f.close()
total_raw.append(raw)

raw = []
f = open("3.out", "r")
log = False
one_iter = []
while True:
    line = f.readline()
    if not line: break
    if "_" not in line and not log:
        log = True
        if len(one_iter) != 0:
            raw.append(one_iter)
        one_iter = []
    elif "_" in line and log:
        log = False

    if log:
        one_iter.append(float(line.strip()))
f.close()
total_raw.append(raw)


print(len(total_raw[0]))
for i, raw in enumerate(total_raw):
    last_iter = raw[-1]
    print("{:.6f}".format(sum(last_iter) / len(last_iter)))
