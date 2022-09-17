import csv
import math
import datetime
import pandas as pd
from datetime import date
from matplotlib import pyplot
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore")

input_data = '../COVID19_data.csv'
use_cols = ['Date','Confirmed','Tested','First Dose Administered']
vaccine=[]
tests=[]
realCases=[]
averageTests=[]
alpha=1/5.8
gamma=1/5
eta=0.66
N=70000000
actualY=[]

def readCsv(filename):
	return pd.read_csv(filename, sep=',',usecols=use_cols)

def lossFunction(actualY,predictedY):
	loss=0
	for i in range(len(actualY)):
		if(predictedY[i]<=0):
			predictedY[i]=1
		loss+=(((actualY[i]))-((predictedY[i])))**2
	return loss/42


def gradientEstimate(beta,s0,e0,i0,r0,cir0):
	
	predictedY=generatePredictedY(beta+0.01,s0,e0,i0,r0,cir0)
	loss1=lossFunction(actualY,predictedY)
	predictedY=generatePredictedY(beta-0.01,s0,e0,i0,r0,cir0)
	loss2=lossFunction(actualY,predictedY)
	partialBeta=(loss1-loss2)/(2*0.01)

	predictedY=generatePredictedY(beta,s0+1,e0,i0,r0,cir0)
	loss1=lossFunction(actualY,predictedY)
	predictedY=generatePredictedY(beta,s0-1,e0,i0,r0,cir0)
	loss2=lossFunction(actualY,predictedY)
	partialS0=(loss1-loss2)/(2)

	
	predictedY=generatePredictedY(beta,s0,e0+1,i0,r0,cir0)
	loss1=lossFunction(actualY,predictedY)
	predictedY=generatePredictedY(beta,s0,e0-1,i0,r0,cir0)
	loss2=lossFunction(actualY,predictedY)
	partialE0=(loss1-loss2)/(2)

	
	predictedY=generatePredictedY(beta,s0,e0,i0+1,r0,cir0)
	loss1=lossFunction(actualY,predictedY)
	predictedY=generatePredictedY(beta,s0,e0,i0-1,r0,cir0)
	loss2=lossFunction(actualY,predictedY)
	partialI0=(loss1-loss2)/(2)


	predictedY=generatePredictedY(beta,s0,e0,i0,r0+1,cir0)
	loss1=lossFunction(actualY,predictedY)
	predictedY=generatePredictedY(beta,s0,e0,i0,r0-1,cir0)
	loss2=lossFunction(actualY,predictedY)
	partialR0=(loss1-loss2)/(2)

	
	predictedY=generatePredictedY(beta,s0,e0,i0,r0,cir0+0.1)
	loss1=lossFunction(actualY,predictedY)
	predictedY=generatePredictedY(beta,s0,e0,i0,r0,cir0-0.1)
	loss2=lossFunction(actualY,predictedY)
	partialCir0=(loss1-loss2)/(2*0.1)

	return (partialBeta),(partialS0),(partialE0),(partialI0),(partialR0),(partialCir0)

#generates model output for first 42 days
def generatePredictedY(beta,s0,e0,i0,r0,cir0):
	t=42
	S=[]
	E=[]
	I=[]
	R=[]
	S.append(s0)
	E.append(e0)
	I.append(i0)
	R.append(r0)
	predictedY=[]
	for i in range(t):
		deltaVaccine=vaccine[i+1]-vaccine[i]
		if(i>=30):
			deltaW=0
		else:
			deltaW=r0/30
		
		S.append((-beta*S[i])*(I[i]/N) - eta*deltaVaccine + deltaW+S[i])
		E.append((beta*S[i])*(I[i]/N) - alpha*E[i]+E[i])
		I.append(alpha*E[i] - gamma*I[i]+I[i])
		R.append(gamma*I[i]+eta*deltaVaccine - deltaW+R[i])
	cirT=[]
	testedInitial=averageTests[0]
	for i in range(len(averageTests)):
		cirT.append(cir0*(testedInitial/averageTests[i]))
	
	deltaI=[]

	for i in range(t):
		deltaI.append(E[i]/cirT[i])

	for i in range(t):
		s=0
		c=0
		for j in range(max(i-6,0),i+1):
			c+=1
			s+=deltaI[j]
		if((s/c)<=0):
			predictedY.append(0)
		else:
			predictedY.append(math.log((s/c)*alpha))

	return predictedY


def gradientDescent(n_iter):

	#initial guess on the basis of some analysis and trial and error
	share=0.79699+0.20
	e0=0.002*N
	i0=0.00101*N
	minl=999999
	r0F=0.156
	mr=99999
	grad=0
	s0F=share-r0F
	while(r0F<=0.2):
		r0F+=0.001
		s0F=share-r0F
		r0=r0F*N
		s0=s0F*N
		cir0=20.9
		while(cir0<=22):
			cir0+=0.1
			beta=random.random()
			for i in range(n_iter):
				predictedY=generatePredictedY(beta,s0,e0,i0,r0,cir0)
				loss=lossFunction(actualY,predictedY)
				betaD,s0D,e0D,i0D,r0D,cir0D = gradientEstimate(beta,s0,e0,i0,r0,cir0)
				if(loss<=0.01):
					return (beta,s0,e0,i0,r0,cir0,loss)
				if(loss<minl):
					mr=r0F
					grad=betaD+s0D+e0D+i0D+r0D+cir0D
					minl=loss
					bbeta,ss0,ee0,ii0,rr0,ccir0=(beta,s0,e0,i0,r0,cir0)
				beta=beta - betaD/(i+1)
				if(beta>1 or beta<0):
					break
				s0=s0 - s0D/(i+1)
				e0=e0 - e0D/(i+1)
				i0=i0 - i0D/(i+1)
				r0=r0 - r0D/(i+1)
				if(r0<0.156*N or r0>0.360):
					break
				cir0=cir0 - cir0D/(i+1)
				if(cir0<12 or cir0>30):
					break
	return (bbeta,ss0,ee0,ii0,rr0,ccir0,minl)

def generateActualCasesData(data):
	testedInitial=0
	actualY.clear()
	realCases.clear()
	c=0
	lol=0
	for i in range(len(data)):
		if(data['Date'][i]>='2021-03-16' and data['Date'][i]<='2021-04-26'):
			
			vaccine.append(data['First Dose Administered'][i])
			if(data['Date'][i]=='2021-03-16'):
				for j in range(i-7,i):
					tests.append(data['Tested'][j])
					lol+=data['Tested'][j]
				lol/=7
			tests.append(data['Tested'][i])
			deltaConfirmedAvg=0
			tt=0
			for j in range(i-6,i+1):
				tt+=data['Tested'][j-1]



			deltaConfirmedAvg=(data['Confirmed'][i]-data['Confirmed'][i-7])
			
			deltaConfirmedAvg/=7
			tt/=7
			
			actualCases=(deltaConfirmedAvg)
			realCases.append((deltaConfirmedAvg))
			actualY.append(math.log(actualCases))
			
		if(data['Date'][i]>'2021-04-26'):
			tests.append(data['Tested'][i])
			vaccine.append(data['First Dose Administered'][i])
			tt=0
			for j in range(i-6,i-1):
				tt+=data['Tested'][j-1]
				deltaConfirmedAvg+=(data['Confirmed'][j+1]-data['Confirmed'][j])
			tt/=7
			deltaConfirmedAvg/=7
			realCases.append((deltaConfirmedAvg))
	for i in range(7,len(tests)):
		s=0
		for j in range(i-7,i):
			s+=tests[i]
		averageTests.append(s/7)

def manualEx():
	# predictedY=generatePredictedY(0.5162831695074731, 58029300.0, 140000.00000155042, 69299.99999765515, 11900000.0,12.899999999999997)
	# x=[]
	# for i in range(42):
	# 	x.append(i)
	# plt.scatter(x,actualY)
	# plt.scatter(x,predictedY)
	# plt.show()

	# predictedY=generatePredictedY(0.4264605057922473, 58799298.99999987, 139998.99998796944, 70698.99998223415, 10989999.99999992, 26.92174376836325)
	# print(lossFunction(actualY,predictedY))

	# predictedY=generatePredictedY(0.3130068090413153, 58799298.0, 139998.0, 70698.0, 10990006.0, 26.2437254391853)
	# print(lossFunction(actualY,predictedY))
	beta=0.484510183197277
	s0=58029300.0
	e0=140000.0
	i0=70700.0
	r0=11760000.0
	cir0=21.400000000000006
	predictedY=generatePredictedY(beta ,s0 ,e0,i0,r0,cir0)
	loss=(lossFunction(actualY,predictedY))

	print('Best LOSS: ',loss)
	print('Best Fit Params: ','Beta: ',beta,' S0: ',s0,' E0: ',e0,' I0: ',i0,' R0: ',r0,' CIR0: ',cir0)
	

def runPredictor(t,beta,s0,e0,i0,r0,cir0,loop):
	S=[]
	E=[]
	I=[]
	R=[]
	S.append(s0)
	E.append(e0)
	I.append(i0)
	R.append(r0)
	mainBeta=beta
	newI=[]
	predictedY=[]
	for i in range(t):
		if(i<=41):
			deltaVaccine=vaccine[i+1]-vaccine[i]
		else:
			deltaVaccine=200000
		if(i>=30 and i<=180):
			deltaW=0
		elif(i<30):
			deltaW=r0/30
		else:
			deltaVaccineAgg=0
			if(i-179<=41):
				deltaVaccineAgg=abs(vaccine[i-179]-vaccine[i-180])
			else:
				deltaVaccineAgg=200000
			rAdd=R[i-179]-R[i-180]
			if(rAdd<0):
				rAdd=0
			deltaW=rAdd+eta*deltaVaccineAgg
			if(deltaW>=R[i]+gamma*I[i] +delv):
				deltaW=R[i]+gamma*I[i] +delv
		if (i<188):
			testedInitial=averageTests[0]	
			cirT=(cir0*(testedInitial/averageTests[i]))

		delv=eta*deltaVaccine
		delW=deltaW
		if(S[i]+delW<delv):
			delv=0
		
		delW=deltaW
		S.append((-beta*S[i])*(I[i]/N) +S[i]-delv+delW)
		E.append((beta*S[i])*(I[i]/N) - alpha*E[i]+E[i])
		I.append(alpha*E[i] - gamma*I[i]+I[i])
		newI.append(alpha*E[i]/cirT)
		R.append(gamma*I[i] +R[i]+delv-delW)

		if(S[i]<0 or S[i]>N):
			print('Serr')
		if(E[i]<0 or E[i]>N):
			print('Eerr')
		if(I[i]<0 or I[i]>N):
			print('Ierr')
		if(R[i]<0 or R[i]>N):
			print('Rerr')
		
		
		if (loop=='close'):
			if ((i)%7==0):
				
				avgNewCases=0
				c=0
				for j in range(max(i-6,0),i+1):
					c+=1
					avgNewCases+=newI[j]
				avgNewCases/=c
				if(avgNewCases<=10000):
					beta=mainBeta
				elif(avgNewCases>=10001 and avgNewCases<=25000):
					beta=(2*mainBeta)/3
				elif(avgNewCases>=25001 and avgNewCases<=100000):
					beta=mainBeta/2
				else:
					beta=mainBeta/3

	for i in range(len(S)):
		S[i]/=N
	return newI,S

def openLoopControl(runTime,beta,s0,e0,i0,r0,cir0):
	x=[]
	date = datetime.datetime(2021,3,16)
	for i in range(runTime):
		x.append(date)
		date += datetime.timedelta(days=1)
	Y,S=runPredictor(runTime,beta,s0,e0,i0,r0,cir0,loop='open')
	plt.xlabel('Date')
	plt.ylabel('New Cases Per Day, Beta')
	plt.title('Daily New Cases, Open Loop for Beta')
	plt.xticks(rotation=90)
	plt.plot(x,Y,label='predicted')
	plt.plot(x,realCases,label='actual')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

	plt.xlabel('Date')
	plt.title('Variation in Susceptible Fraction in Open Loop for Beta')
	plt.ylabel('Total Susceptible Fraction, Beta')
	plt.xticks(rotation=90)
	plt.plot(x,S[0:runTime],label='Susceptible Fraction')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()
	
	Y,S=runPredictor(runTime,2*beta/3,s0,e0,i0,r0,cir0,loop='open')
	plt.xlabel('Date')
	plt.ylabel('New Cases Per Day, 2*Beta/3')
	plt.title('Daily New Cases, Open Loop for 2*Beta/3')
	plt.xticks(rotation=90)
	plt.plot(x,Y,label='predicted')
	plt.plot(x,realCases,label='actual')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

	plt.xlabel('Date')
	plt.title('Variation in Susceptible Fraction in Open Loop for 2*Beta/3')
	plt.ylabel('Total Susceptible Fraction, 2*Beta/3')
	plt.xticks(rotation=90)
	plt.plot(x,S[0:runTime],label='Susceptible Fraction')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

	Y,S=runPredictor(runTime,beta/2,s0,e0,i0,r0,cir0,loop='open')
	plt.xlabel('Date')
	plt.ylabel('New Cases Per Day, Beta/2')
	plt.title('Daily New Cases, Open Loop for Beta/2')
	plt.xticks(rotation=90)
	plt.plot(x,Y,label='predicted')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

	plt.xlabel('Date')
	plt.title('Variation in Susceptible Fraction in Open Loop for Beta/2')
	plt.ylabel('Total Susceptible Fraction, Beta/2')
	plt.xticks(rotation=90)
	plt.plot(x,S[0:runTime],label='Susceptible Fraction')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

	Y,S=runPredictor(runTime,beta/3,s0,e0,i0,r0,cir0,loop='open')
	plt.xlabel('Date')
	plt.ylabel('New Cases Per Day, Beta/3')
	plt.title('Daily New Cases, Open Loop for Beta/3')
	plt.xticks(rotation=90)
	plt.plot(x,Y,label='predicted')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()	

	plt.xlabel('Date')
	plt.title('Variation in Susceptible Fraction in Open Loop for Beta/3')
	plt.ylabel('Total Susceptible Fraction, Beta/3')
	plt.xticks(rotation=90)
	plt.plot(x,S[0:runTime],label='Susceptible Fraction')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

def closedLoopControl(runTime,beta,s0,e0,i0,r0,cir0):
	x=[]
	date = datetime.datetime(2021,3,16)
	for i in range(runTime):
		x.append(date)
		date += datetime.timedelta(days=1)
	Y,S=runPredictor(runTime,beta,s0,e0,i0,r0,cir0,loop='close')
	plt.xlabel('Date')
	plt.title('Daily New Cases in Closed Loop')
	plt.ylabel('New Cases Per Day, Closed Loop')
	plt.xticks(rotation=90)
	plt.plot(x,Y)
	plt.tight_layout()
	plt.show()

	plt.title('Variation in Susceptible Fraction in Closed Loop')
	plt.xlabel('Date')
	plt.ylabel('Total Susceptible Fraction, Closed Loop')
	plt.xticks(rotation=90)
	plt.plot(x,S[0:runTime],label='Susceptible Fraction')
	plt.tight_layout()
	plt.legend(loc="upper right")
	plt.show()

if __name__ == '__main__':
	runTime=500
	data = readCsv(input_data)
	
	generateActualCasesData(data)
	
	beta,s0,e0,i0,r0,cir0,loss=gradientDescent(n_iter=100000)
	print('Optimal Params From Report: ')
	manualEx()
	print('\n\n\n')
	print('Fresh set of optimal Params:')
	print('Best LOSS: ',loss)
	print('Best Fit Params: ','Beta: ',beta,' S0: ',s0,' E0: ',e0,' I0: ',i0,' R0: ',r0,' CIR0: ',cir0)
	predictedY=generatePredictedY(beta,s0,e0,i0,r0,cir0)
	remain = runTime-len(realCases)
	
	for i in range(remain):
		realCases.append(0)

	#Running predictions for next 500 days - Uncomment to verify plots
	# openLoopControl(runTime,beta,s0,e0,i0,r0,cir0)
	# closedLoopControl(runTime,beta,s0,e0,i0,r0,cir0)






