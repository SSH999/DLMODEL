n = int(input())

a, b = 0, 1
for i in range(2, n+1):
    c = (a + b) % 10
    a = b
    b = c

print(b)
n = int(input())

a, b = 0, 1
for i in range(2, n+1):
    c = (a + b) % 10
    a = b
    b = c

print(b)
