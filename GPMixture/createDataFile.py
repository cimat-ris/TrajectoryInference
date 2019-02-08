# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 23:18:53 2016

@author: karenlc
"""

#Programa para escribir los nombres de las trayectorias en un archivo
"""
f = open("pathSet1000.txt","w")
s = "0000"

for i in range(1000):
    if i < 9:
        s = "Data/00000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 9 and i < 99:
        s = "Data/0000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 99 and i < 999:
        s = "Data/000%d.txt\n"%(i+1)
        f.write(s)  
s = "Data/00%d.txt\n"%(1000)   
f.write(s) 

f.close()

f = open("pathSet100.txt","w")
s = "0000"

for i in range(100):
    if i < 9:
        s = "Data/00000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 9 and i < 99:
        s = "Data/0000%d.txt\n"%(i+1)
        f.write(s)  
s = "Data/000%d.txt\n"%(100)   
f.write(s) 

f.close()

f = open("pathSet400.txt","w")
s = "0000"

for i in range(1000):
    if i < 9:
        s = "Data/00000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 9 and i < 99:
        s = "Data/0000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 99 and i < 400:
        s = "Data/000%d.txt\n"%(i+1)
        f.write(s)  
f.close()

f = open("pathSet2000.txt","w")
s = "0000"

for i in range(2000):
    if i <= 9:
        s = "Data/00000%d.txt\n"%(i+1)
        f.write(s)   
    if i > 9 and i <= 99:
        s = "Data/0000%d.txt\n"%(i+1)
        f.write(s)   
    if i > 99 and i <= 999:
        s = "Data/000%d.txt\n"%(i+1)
        f.write(s)   
    if i > 999 and i <= 2000:
        s = "Data/00%d.txt\n"%(i+1)
        f.write(s)  
f.close()
"""

f = open("pathSet10000.txt","w")
#s = "0000"
for i in range(10000):
    if i < 9:
        s = "Data/00000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 9 and i < 99:
        s = "Data/0000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 99 and i < 999:
        s = "Data/000%d.txt\n"%(i+1)
        f.write(s)   
    if i >= 999 and i < 9999:
        s = "Data/00%d.txt\n"%(i+1)
        f.write(s) 
    if i == 9999:
        s = "Data/0%d.txt\n"%(i+1)
        f.write(s)  

f.close()





