Class = {}

while True:
    name = input("Enter Studen's Name and press enter for stop")
    if name == '':
        break
    score = int(input("Enter Scores: "))
    if name in Class:
        Class[name] += (score,)
    else:
        Class[name] = (score,)
for name in sorted(Class.keys()):
    sum = 0
    count = 0
    for score in Class[name]:
        sum += score
        count += 1
    print(sum)
    print(name, ":", sum / count)

