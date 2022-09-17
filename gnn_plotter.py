import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
import re
import numpy as np
numMetrics=5
numExperiments=5
mx_val=9999999999999999
model='gcn'

#y_amazon = [756385.0, 440718.0, 266359.0]
#y_reddit = [184307.0, 99544.0, 95344.0]
#y_arxiv = [59998.0, 38744.0, 18673.0]

#x_arxiv = [165,82,41]
#x_reddit = [227,113,56]
#x_amazon = [2391,1195,597]
def violins(num_msgs,barrier_syncs):
	msgs_data= [[],[],[],[],[],[],[],[]]
	barrier_syncs_data=[[],[],[],[],[],[],[],[]]
	for i in range(len(msg_counts)):
		msgs_data[i%nss].append(msgs_data[i])
		barrier_syncs_data[i%nss].append(barrier_syncs[i])
	norm=[0,1,2,3,4,5,6,7]

def quick_plot():
	y_amazon = [756385.0, 440718.0, 266359.0]
	y_reddit = [184307.0, 99544.0, 95344.0]
	y_arxiv = [59998.0, 38744.0, 18673.0]

	x_arxiv = [165,82,41]
	x_reddit = [227,113,56]
	x_amazon = [2391,1195,597]

	plt.title("Total Giraph Overheads Trend for batch sizes 4096, 2048 and 1024 over 3 Graphs")
	plt.xlabel("# of Minibatches")
	plt.ylabel("time(s)")
	#plt.ytick([])
	plt.plot(x_arxiv,np.array(y_arxiv)/1000,color='green',label='arxiv',marker='o')
	# plt.legend()
	# plt.show()
	# plt.title("Total Giraph Overheads Trend for batch sizes 4096, 2048 and 1024 over 3 Graphs")
	# plt.xlabel("# of Minibatches")
	# plt.ylabel("time(s)")
	plt.plot(x_reddit, np.array(y_reddit)/1000,color='red',label='reddit',marker='x')
	# plt.legend()
	# plt.show()
	# plt.title("Total Giraph Overheads Trend for batch sizes 4096, 2048 and 1024 over 3 Graphs")
	# plt.xlabel("# of Minibatches")
	# plt.ylabel("time(s)")
	plt.plot(x_amazon,np.array(y_amazon)/1000,color='yellow',label='products',marker='D')
	plt.legend()
	plt.show()



#######################################
#AMAZON16 c1024.txt = corrupt
########################## 

tick_labels=['' ,1024,'' ,2048,'',4096, '',4096]

#archive
# graph="arxiv"
# batch_sizes=[1024,2048,4096]

# nodes=169343
# alias="arxiv"
# ytick = [0,20,40,60,80,100,120,140,160,180,200,220]
# ytick2= [3000000,3500000,4000000]

# #amazon
# graph="/amazon"
# batch_sizes=[1024,2048,4096]
# nodes=2449029
# alias="amazon"
# ytick = [0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400]
# ytick2 = [250000000,300000000,350000000]

#reddit
graph="/reddit"
batch_sizes=[1024,2048,4096]
nodes= 232965
alias="reddit"
ytick = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]
ytick2 = [10000000,15000000,20000000,25000000,30000000]

# files_dir = '/Users/varad.kulkarni/hadoop-3.1.1/test'
# batch_sizes=[4096]

nss=8

def get_gc_times(lines):
	matches = re.findall(r'time spent on gc was (\d+|\d+\.\d+) s', lines)

	for i in range(len (matches)):
		matches[i]=float(matches[i])
	return matches

def match_string_for_msgCount(lines):
	substring = 'BspServiceMaster.java:aggregateWorkerStats(992)'
	out_list=[]
	for line in lines.split('\n'):
		if(substring in line):
			out_list.append(line)

	return out_list

msgCount="msgCount="
def get_msg_count(s):
	ind=0
	for i in range(len(s)):
		strr=s[i:i+len(msgCount)]
		if strr==msgCount:
			ind=i+len(msgCount)
			break
	dig=''
	while s[ind].isdigit():
		dig+=s[ind]
		ind+=1
	return (int(dig))


def get_all_matches_per_worker(lines):
	matches = re.findall(r'LOG FOR G2N2: (STARTING COMPUTE|ENDED COMPUTE|ALL MESSAGES FLUSHED|BARRIER SYNC REACHED) IN SS: (\d+) AT TIME: (\d+|\d+\.\d+)', lines)
	return matches

def get_all_matches_for_master(lines):
	matches = re.findall(r'Coordination of superstep (\d+) took (\d+|\d+\.\d+) seconds', lines)
	ints=[]
	for match in matches:
		ints.append(float(match[1])*1000)
	return ints
####################################INCOMPLETE FNCTN FOR TESTING##########################



def get_file_list(files_dir):
	
	for i in range(len(batch_sizes)):
		totalSS.append((nodes//batch_sizes[i])*nss)
	print(totalSS)
	key =  files_dir
	files = subprocess.run(['ls', key+'/'], stdout=subprocess.PIPE)
	files = files.stdout.decode().split()[:len(batch_sizes)]
	files = [file for file in files if '.txt' in file]
	print(files)
	return files

#############################################################################################

def plot_og_bars(superstep_times,exclusive_computes,exclusive_comm,exclusive_sync_costs,overheads,batch_size):
	y=[[],[],[],[],[],[]]

	for i in range(numMetrics+1):
		for j in range(nss):
			y[i].append(0)

	for i in range(len(superstep_times)):
		y[0][i%nss]+=superstep_times[i]
		y[1][i%nss]+=exclusive_computes[i]
		y[2][i%nss]+=exclusive_comm[i]
		y[3][i%nss]+=exclusive_sync_costs[i]
		y[4][i%nss]+=overheads[i]
		y[5][i%nss]+=overheads[i]+exclusive_sync_costs[i]+exclusive_comm[i]
	
	wc=int(xx)
	x = np.arange(8)
	width = 0.1
	plt.bar(x-0.3, y[0], width, color='red',edgecolor='black')
	plt.bar(x-0.2, y[1], width, color='orange',edgecolor='black')
	plt.bar(x-0.1, y[2], width, color='pink',edgecolor='black')
	plt.bar(x+0.1, y[3], width, color='yellow',edgecolor='black')
	plt.bar(x+0.2, y[4], width, color='gold',edgecolor='black')
	plt.bar(x+0.3, y[5], width, color='blue',edgecolor='black')
	

	plt.title("Total Time For wc="+str(wc)+", bs="+str(batch_size))
	plt.xticks(x, ['0_CT', '1_CT', '2_CT+FPInit', '3_FP', '4_FPEnd','5_BPInit','6_BP+WakeAll','7_ClearAll'])
	plt.xlabel("SuperStepNumber_ComputationPhase(CT=ComputationTree, FP=FwdPass, BP=BackPass) ")
	plt.ylabel("Time(ms)")
	plt.legend(["SS Time", "Exclusive Compute Time","Exclusive Communication Time" ,"Exclusive Synchronisation Costs","Constant Overheads","TotOverheads=SS_Time-ExclusiveComputeTime"])
	plt.grid()
	#plt.yticks([0,50,100,150,200,250,300,350])
	plt.xticks(rotation=0)
	plt.show()

def plot_lines(totalSS,tot_ss,tot_cp,tot_cm,tot_bs,tot_oh):
	totalMinibatches=[]
	for i in range(len(totalSS)):
		totalMinibatches.append(totalSS[i]//nss)

	total_over=tot_cm+tot_bs+tot_oh
	plt.title("Total time per superstep vs NumOfMinibatches, batch sizes 4096, 2048 and 1024, graph= " +alias+", wc="+str(xx)+',model='+model)
	plt.xlabel("# of Minibatches")
	plt.ylabel("time(s)")

	plt.plot(totalMinibatches,tot_ss/1000,label='ss_times',marker='o')
	plt.plot(totalMinibatches,tot_cp/1000,label='exclusive_compute_time',marker='o')
	plt.plot(totalMinibatches,tot_cm/1000,label='exclusive_communication_time',marker='o')
	plt.plot(totalMinibatches,tot_bs/1000,label='exclusive_barrier_synch_time',marker='o')
	plt.plot(totalMinibatches,tot_oh/1000,label='Constant overheads',marker='o')
	plt.plot(totalMinibatches,total_over/1000,label='TotOverheads=SS_Time-ExclusiveComputeTime',marker='o')
	plt.yticks(ytick)
	plt.grid()
	plt.legend()
	plt.show()
	

def plot_new_bars_and_stacks(sup_step_list,compute_time_list,comm_time_list,barrier_synch_list,overheads_list,total_msg,gc_total):

	print("X0X0X0X0X0XX0-----GCCCCC")
	print(gc_total)
	features = ['superstep times', 'exclusive_compute_time', 'network communication time','barrier_sync', 'overheads' ]
	total_times = [sup_step_list,compute_time_list,comm_time_list,barrier_synch_list,overheads_list]
	#print(len(sup_step_list), np.array([np.sum(sup_step_list[0])]).shape)
	total_times = [np.stack(np.array([np.sum(a) for a in n])) for n in total_times]
	#print([n.shape for n in total_times])
	total_times = np.stack([a for a in total_times])
	
	width = 0.2   # the width of the bars: can also be len(x) sequence
	labels = ['','0_CT', '1_CT', '2_CT+FPInit', '3_FP', '4_FPEnd','5_BPInit','6_BP+WakeAll','7_ClearAll']
	colors = ['red', 'blue', 'orange', 'pink',  'cyan', 'green']
	ecolors = ['black', 'pink']
	fig, ax = plt.subplots()
	
	x = np.arange(total_times.shape[1])

	ax.bar(x-0.1, total_times[0, :]/1000, width, color = colors[0], ecolor = ecolors[0])
	for i in range(1, total_times.shape[0]):
		ax.bar(x+0.1, total_times[i, :]/1000, width, \
				bottom=np.sum(total_times[1:i, :]/1000, axis=0), color = colors[i], ecolor = ecolors[i%2])

	ax.set_ylabel('time (s)')
	ax.set_xlabel("Batch size ")
	ax.set_xticklabels(tick_labels)
	# ax2 = ax.twinx()
	# ax2.set_ylabel('Total Number of Msgs',color='green')
	# ax2.plot(x,total_msgs,color='green')
	# ax2.set_yticklabels(de,color='green')
	# ax2.set_yticks(ytick2)
	
	de=[]
	for i in range(len(ytick2)):
		de.append(str(ytick2[i]))
	lols=[]
	for i in range (len(ytick)):
		lols.append(str(ytick[i]))

	
	ax.set_yticklabels(lols)
	ax.set_yticks(ytick)
	#ax.figure.set_size_inches(20,12)
	ax.set_title("Total time Bucket Split Per Epoch vs Batch Size, graph= " +alias+" for wc="+str(xx))
	ax.legend(features)
	ax.grid(axis='y', linewidth=0.3)
	#plt.savefig('stackedBars_phaseSum.jpg',  bbox_inches='tight')
	plt.show()
	return 0



def first_msg_times(lines):
	matches=re.findall(r'SENDING FIRST MSG IN SS:(\d+)AT TIME: (\d+|\d+\.\d+)',lines)
	return matches

def convert(tms,ss):
	init_times=np.full(ss+5,mx_val)
	
	for tm in tms:
		a,b=int(tm[0]),int(tm[1])
		init_times[a]=min(init_times[a],b)
	
	for i in range(len(init_times)):
		if init_times[i] == mx_val:
			init_times[i]=0
	
	return init_times

def minus(a,b,exclusive_computes):
	ans=np.full(len(b),0)
	for i in range(len(b)):
		if a[i]!=0:
			ans[i]=a[i]-b[i]
		else:
			ans[i]=exclusive_computes[i]
	return ans

def minnus(a,b):
	ans=np.full(len(a),0)
	for i in range(len(a)):
		if(b[i]!=0):
			ans[i]=a[i]-b[i]
	return ans

def viol(e1,e2,batch):
	
	# ax2.set_yticklabels(de,color='green')
	# ax2.set_yticks(ytick2)
	y2=[331809346,332451028]
	#ytick2 = [250000000,300000000,350000000]
	exclusive_comms=[e1,e2]
	fig, ax = plt.subplots()
	
	ax.set_title("Exclusive Comm. Time, graph= " +alias+", wc=8 and 16,model="+model+' batch Size='+(batch))
	ax.set_xlabel("Distribution of Exclusive Communication Time across All SS")
	ax.set_xticks([0,1,2])

	ax.set_xticklabels(['','wc=8','wc=16'])
	ax.set_ylabel("time(s)")
	ax.violinplot(exclusive_comms)
	ax.grid()


	x=[1,2]
	ax2 = ax.twinx()
	# ax2.grid(color='green')
	
	# ax2.set_yticklabels(['','0','100000000','200000000','322812228','400000000'])
	ax2.set_ylim(0,4*1e8)
	# ax2.set_yticks([0,3*1e8,4*1e8])
	ax2.set_ylabel('Total Number of Msgs',color='green')
	ax2.plot(x,y2,color='green')
	plt.show()

def plot_exc_comm(ss,exclusive_comm, batch):
	yy=[0,100,200,300,400,500,600]
	ytick = ['0','100','200','300','400','500','600']
	plt.title("Exclusive Comm. Time, graph= " +alias+", wc="+str(xx)+',model='+model+' batch Size='+(batch))
	plt.xlabel("SS Number")
	plt.ylabel("time(s)")

	a=np.arange(ss)
	plt.plot(a,exclusive_comm)
	plt.yticks(yy)
	plt.grid()
	plt.show()

	

if __name__ == '__main__':
	#quick_plot()
	e1=[]
	for vk in range(0,numExperiments):
		totalSS=[]
		ws=2**vk ##worker count
		xx=str(ws)
		
		print(xx+" WORKERS!!")
		print("XXXXXXXXXXXXXXXXXX----------------XXXXXXXXXXXXXXXXXX")
		files_dir = '/Users/varad.kulkarni/hadoop-3.1.1/gnn/'+model+'/'+graph+'/'+xx
		# files_dir = '/Users/varad.kulkarni/hadoop-3.1.1/test' #add sample.txt in /test
		# batch_sizes=[4096]
		# ws=1
		# xx="1"
		tot_times=[]
		files = get_file_list(files_dir)
		print(files)
		kk=0
		all_times_list=[]
		total_times=[]
		tot_ss=[]
		tot_cp=[]
		tot_cm=[]
		tot_bs=[]
		tot_oh=[]
		total_msgs=[]
		sup_step_list=[]
		total_compute_time_list=[]
		comm_time_list=[]
		barrier_synch_list=[]
		overheads_list=[]
		gc_total=[]
		
		
		for file in files:
			bc=file[1:5]
			if file!='a1024.txt':
				kk+=1
				continue
			lol=totalSS[kk]%nss
			lol=totalSS[kk]-lol
			lol+=1
			print(lol-1)
			fname = files_dir+'/'+file
			f = open(fname, "r")
			lines=f.read()
			print('Read: '+fname)
			metrics=match_string_for_msgCount(lines)
			msg_counts=[]

			for i in range(2,lol+1):
				msg_counts.append(get_msg_count(metrics[i]))

			print(np.sum(np.array(msg_counts)))
			first_times=first_msg_times(lines)
			converted_first_times=convert(first_times,lol-1)
			print('hello')
			print(len(converted_first_times))
			
			gc_times = get_gc_times(lines)
			gc_total.append(np.sum(gc_times))
			matches_per_worker=get_all_matches_per_worker(lines)
			all_compute_starts=[]
			all_compute_ends=[]
			all_barrier_syncs=[]
			all_msgs_flushed=[]
			for match in ((matches_per_worker)):
				if match[0]=='STARTING COMPUTE':
					all_compute_starts.append(int(match[2]))
				elif match[0]=='ENDED COMPUTE':
					all_compute_ends.append(int(match[2]))	
				elif match[0]=='ALL MESSAGES FLUSHED':
					all_msgs_flushed.append(int(match[2]))
				elif match[0]=='BARRIER SYNC REACHED':
					all_barrier_syncs.append(int(match[2]))	


			superstep_times=np.array(get_all_matches_for_master(lines))
			# print(len(all_compute_starts))
			# print(len(all_compute_ends))
			# print(len(all_msgs_flushed))
			# print(len(all_barrier_syncs))
			# print(len(superstep_times))

			all_compute_starts=np.array(all_compute_starts)
			all_compute_ends=np.array(all_compute_ends)
			all_msgs_flushed=np.array(all_msgs_flushed)
			all_barrier_syncs=np.array(all_barrier_syncs)

			min_compute_start_times=[]
			max_compute_end_times=[]
			max_msg_flushed=[]
			max_barrier_syn=[]
			wc=int(xx)
			total_ss=len(all_compute_starts)//wc
			for i in range(total_ss):
				tmp1=[]
				for j in range(wc):
					tmp1.append(all_compute_starts[total_ss*j+i])
				min_compute_start_times.append(min(tmp1))
				tmp2=[]
				for j in range(wc):
					tmp2.append(all_compute_ends[total_ss*j+i])
				max_compute_end_times.append(max(tmp2))
				tmp3=[]
				for j in range(wc):
					tmp3.append(all_msgs_flushed[total_ss*j+i])
				max_msg_flushed.append(max(tmp3))
				tmp4=[]
				for j in range(wc):
					tmp4.append(all_barrier_syncs[total_ss*j+i])
				max_barrier_syn.append(max(tmp4))


			min_compute_start_times=(np.array(min_compute_start_times))
			max_compute_end_times=(np.array(max_compute_end_times))
			min_first_msg_times=converted_first_times
			max_msg_flushed=(np.array(max_msg_flushed))
			max_barrier_syn=(np.array(max_barrier_syn))

			
			comp_plus_comm = minnus(max_compute_end_times, min_first_msg_times)
			exclusive_computes = max_compute_end_times- min_compute_start_times
			actual_exclusive_compute = minus(min_first_msg_times,min_compute_start_times,exclusive_computes)
			exclusive_comm = max_msg_flushed - max_compute_end_times
			exclusive_sync_costs = max_barrier_syn - max_msg_flushed

			# print(actual_exclusive_compute[1:13])
			# print(comp_plus_comm[1:13])
			# print(exclusive_computes[1:13])

			buckets=exclusive_computes+exclusive_comm+exclusive_sync_costs
			overheads = superstep_times-buckets
			
			total_giraph_overheads = exclusive_comm+exclusive_sync_costs+overheads

			actual_exclusive_compute = actual_exclusive_compute[1:lol] ##TODO: INCLUDE
			comp_plus_comm = comp_plus_comm[1:lol] ##ODO:INCLUDE
			total_computes = exclusive_computes[1:lol]  ##Not_actual exclusive compute, MISNOMER

			exclusive_comm = exclusive_comm[1:lol]
			exclusive_sync_costs = exclusive_sync_costs[1:lol]
			overheads = overheads[1:lol]
			superstep_times=superstep_times[1:lol]
			#print(actual_exclusive_compute)

			total_giraph_overheads = total_giraph_overheads[1:lol]
			
			#gc_total=gc_total[1:lol]
			tot_times.append(np.sum(total_giraph_overheads))

			###########INIT DATA FOR AC####################
			sup_step_list.append(superstep_times)
			total_compute_time_list.append(total_computes)
			comm_time_list.append(exclusive_comm)
			barrier_synch_list.append(exclusive_sync_costs)
			overheads_list.append(overheads)
			###############################################
			#print(exclusive_sync_costs[0:32])
			#plot_og_bars(superstep_times,exclusive_computes,exclusive_comm,exclusive_sync_costs,overheads,batch_sizes[kk])
			tot_ss.append(np.sum(superstep_times))
			tot_cp.append(np.sum(total_computes))
			tot_cm.append(np.sum(exclusive_comm))
			tot_bs.append(np.sum(exclusive_sync_costs))
			tot_oh.append(np.sum(overheads))
			total_msgs.append(np.sum(msg_counts))
			#violins(msg_counts,exclusive_sync_costs)
			if vk==3:
				e1.append(exclusive_comm)

			if vk==4:
				e1.append(exclusive_comm)
			if vk==3 or vk==4:
				if bc=='1024':
					xdk=7
					plot_exc_comm(totalSS[kk],exclusive_comm,bc)

			kk+=1
		
		#plot_lines(totalSS,np.array(tot_ss),np.array(tot_cp),np.array(tot_cm),np.array(tot_bs),np.array(tot_oh))
		if len(e1)==2:
			viol(e1[0],e1[1],'1024')
		#plot_new_bars_and_stacks(sup_step_list,total_compute_time_list,comm_time_list,barrier_synch_list,overheads_list,total_msgs,gc_total)

		# plot_scatter()
		# plot_strong_scaling()
		
		

