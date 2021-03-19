from flask import Flask, render_template
from dataframes.summary import summmary
import numpy as np
from prediction import model
import math
import random

app = Flask(__name__)
x= random.randint(7,15)
total= np.array(summmary())
# p=input('enter how many days back?')

# if p==1:
#       new=prev[3]+x
# elif p==2:
#       new=prev[2]+x
# elif p==3:
#       new=prev[1]+x
# else:
#       new=new
   
   

@app.route('/')
def hello_world():
   prev, new,acc= model()
   return render_template('index.html',total=total,prev=prev,new=abs(new),mse=math.floor(acc))


if __name__ == '__main__':
   app.run(debug=True)