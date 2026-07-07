import numpy as np
import csv, os

def load_csv(path):
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for h in headers:
            data[h] = []
        for row in reader:
            for h, v in zip(headers, row):
                try:
                    data[h].append(float(v))
                except ValueError:
                    data[h].append(0.0)
    for h in headers:
        data[h] = np.array(data[h])
    return data

d3 = load_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "caopengtask1_20260226_1055.csv"))
td = d3["touchdown"]
td_idx = np.where((td[:-1] == 0) & (td[1:] == 1))[0]

i0 = td_idx[12]
i1 = td_idx[29] if len(td_idx) > 29 else len(d3["t_s"])

t_s = d3["t_s"][i0:i1]
t_s = t_s - t_s[0]
vy = d3["v_hat_w1"][i0:i1]

print("vy at 4s:", vy[np.argmin(np.abs(t_s - 4.0))])
print("vy at 5s:", vy[np.argmin(np.abs(t_s - 5.0))])
print("vy at 6s:", vy[np.argmin(np.abs(t_s - 6.0))])
print("vy at 7s:", vy[np.argmin(np.abs(t_s - 7.0))])
print("vy at 8s:", vy[np.argmin(np.abs(t_s - 8.0))])
print("vy at 9s:", vy[np.argmin(np.abs(t_s - 9.0))])
