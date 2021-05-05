f = open('total.txt','w')
for i in range(27661):
    f.write(format(i,'06'))
    f.write('\n')
f.close()
