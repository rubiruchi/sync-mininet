from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.log import setLogLevel, info, output
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.cli import CLI
from mininet.node import Controller, RemoteController
#from mininet.util import custom, waitListening
from collections import defaultdict

from itertools import permutations

import socket, threading, time, errno,copy, struct
import cPickle as pickle
import random
import os.path
import itertools
from dijkstra import *

CoreSwitchList = []
AggSwitchList = []
EdgeSwitchList = []

AllSwitches = []

AllHosts = {}

HostList = []

MassesList = []

MBList = {}

policies = {}
vms = {}
flows = {}
Groups = []
ro = []
masses = {} # to resolve pb of communicating vms hosted on the same server


###############socket ##################
def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
	#len(data)
    return data


########################################

# for graph
#nodes = set()
#edges = {}
#distances = {}
#_distance = []

graphs = {}

def get_graph(graph_name):
  nodes = graphs[graph_name][0]
  edges = graphs[graph_name][1]
  distances = graphs[graph_name][2]
  _distance = graphs[graph_name][3]

def create_graph(graph_name):
  graphs[graph_name] = [None,None,None,None]
  graphs[graph_name][0] = set()
  graphs[graph_name][1] = {}
  graphs[graph_name][2] = {}
  graphs[graph_name][3] = []

def _add_node(value, graph_name):
  graphs[graph_name][0].add(value)

def add_edge(from_node, to_node, distance, graph_name):
  _add_edge(from_node, to_node, distance, graph_name)
  _add_edge(to_node, from_node, distance, graph_name)
  graphs[graph_name][3].append([from_node, to_node, distance])
  graphs[graph_name][3].append([to_node, from_node, distance])

def _add_edge(from_node, to_node, distance, graph_name):
  graphs[graph_name][1].setdefault(from_node, [])
  graphs[graph_name][1][from_node].append(to_node)
  graphs[graph_name][2][(from_node, to_node)] = distance

def dijkstra(initial_node, graph_name):
    nodes = graphs[graph_name][0]
    edges = graphs[graph_name][1]
    distances = graphs[graph_name][2]
    _distance = graphs[graph_name][3]

    
    visited = {initial_node: 0}
    current_node = initial_node
    path = {}

    nodes_ = set(nodes)

    while nodes_:
        min_node = None
        for node in nodes_:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes_.remove(min_node)
        cur_wt = visited[min_node]

        for edge in edges[min_node]:
            wt = cur_wt + distances[(min_node, edge)]
            if edge not in visited or wt < visited[edge]:
                visited[edge] = wt
                path[edge] = min_node

    return visited, path

def shortest_path(initial_node, goal_node, graph_name):

    distances, paths = dijkstra(initial_node, graph_name)
    route = [goal_node]

    while goal_node != initial_node:
        route.append(paths[goal_node])
        goal_node = paths[goal_node]

    route.reverse()
    return route

def get_distance(a, b, graph_name):
    _distance = graphs[graph_name][3]

    for d in _distance:
        if d[0]==a and d[1]==b:
            return d[2]
#exception..

def shortest_distance(src, dst, graph_name):
    '''
    route = shortest_path(src, dst, graph_name)
    d = 0
    for i in range(len(route)-1):
        d = d + get_distance(route[i], route[i+1], graph_name)
    return d
    '''
    cost = CostBetween2Nodes(src,dst)
    return cost

######### get distance between nodes without applying Dijkstra algorithm #########

def IsHost(node):
  if node in HostList:
    return True
  else:
    return False

def IsEdge(node):
  if node in EdgeSwitchList:
    return True
  else:
    return False

def IsAgg(node):
  if node in AggSwitchList:
    return True
  else:
    return False

def IsCore(node):
  if node in CoreSwitchList:
    return True
  else:
    return False

def GetPodId(node):
  k = 14
  num_pod = k  
  num_host = (k/2) 
  num_edge = (k/2) 
  num_agg = (k/2)  
  num_group = k/2  
  num_core = (k/2)

  if IsAgg(node):
    pod = (int(str(node)[1:]) - (num_core*num_group)) / num_agg
  elif IsEdge(node):
    pod = (int(str(node)[1:]) - (num_core*num_group) - (num_pod*num_agg)) / num_edge
  elif IsHost(node):
    pod = (int(str(node)[1:]) - (num_core*num_group) - 2*(num_pod*num_agg)) / (num_host*num_edge)
  else:
    pod = -1

  return pod


def GetCoreGroup(node):
  k = 14
  num_pod = k
  num_host = (k/2)
  num_edge = (k/2)
  num_agg = (k/2)
  num_group = k/2
  num_core = (k/2)

  if IsCore(node):
    group = int(str(node)[1:]) / num_core
  elif IsAgg(node):
    group = (int(str(node)[1:]) - (num_core*num_group)) % num_core
  elif IsEdge(node):
    group = (int(str(node)[1:]) / (num_group*num_core) - (num_pod*num_agg)) % num_core
  elif IsHost(node):
    group = ((int(str(node)[1:]) / (num_group*num_core) - 2*(num_pod*num_agg)) / num_host) % num_core
  else:
    group = -1

  return group

def CostBetween2Nodes(node1, node2):
  #initialise the 3 costs
  costLayer0 = 1
  costLayer1 = 3
  costLayer2 = 5

  # if node1 or node2 are MBs
  if node1 in MBList.keys():
    node1 = MBList[node1][3] 
  if node2 in MBList.keys():
    node2 = MBList[node2][3]

  d = abs(GetCoreGroup(node1)-GetCoreGroup(node2))
  cost = 0
  
  if node1==node2:
    cost = 0
  elif (IsHost(node1) and IsHost(node2)):
    if GetPodId(node1) == GetPodId(node2):
      if (d==0):
        cost = costLayer0 *2
      else:
        cost = 2*costLayer0 + 2*costLayer1
    else:
      cost = 2*costLayer0 + 2*costLayer1 + 2*costLayer2
  elif ((IsHost( node1) and IsEdge( node2)) or (IsHost( node2) and IsEdge(node1))):
    if (GetPodId( node1) == GetPodId( node2)):
      if (d==0):
	cost = costLayer0
      else:
	cost = costLayer0 + 2*costLayer1
    else:
      cost = costLayer0 + 2* costLayer1 + 2*costLayer2
  elif ((IsHost(node1) and IsAgg( node2)) or (IsHost( node2) and IsAgg(node1))):
    if (GetPodId( node1) == GetPodId( node2)):
      cost = costLayer0 + costLayer1
    else:
      cost = costLayer0 + costLayer1 + 2*costLayer2
  elif ((IsHost( node1) and IsCore( node2)) or (IsHost( node2) and IsCore(node1))):
    cost = costLayer0 + costLayer1 + costLayer2
  elif ((IsEdge( node1) and IsAgg( node2)) or (IsEdge( node2) and IsAgg(node1))):
    if (GetPodId( node1) == GetPodId( node2)):
      cost=costLayer1
    else:
      cost=costLayer1 + 2*costLayer2
  elif ((IsEdge( node1) and IsEdge( node2))):
    if (GetPodId( node1) == GetPodId( node2)):
      cost = 2*costLayer1
    else:
      cost = 2*costLayer1 + 2*costLayer2
  elif ((IsEdge( node1) and IsCore( node2)) or (IsEdge( node2) and IsCore( node1)) ):
    cost = costLayer1 + costLayer2
  elif ((IsAgg( node1) and IsCore( node2)) or (IsAgg( node2) and IsCore(node1))):
    if (d==0):
      cost = costLayer2
    else:
      cost = 2*costLayer1 + costLayer2
  elif ((IsAgg( node1) and IsAgg( node2))):
    if (GetPodId( node1) == GetPodId( node2)):
      cost = 2*costLayer1
    else:
      if (d==0):
        cost = 2*costLayer2
      else:
	cost = 2*costLayer1 + 2 * costLayer2
  elif ((IsCore( node1) and IsCore( node2))):
    if (d==0):
      cost = 2*costLayer2
    else:
      cost = 2*costLayer1 + 2*costLayer2

  return cost

   

##################################################################################



def create_graph_flow(graph_id,mbt1,mbt2,mbt3):
  #source=0
  #sink=1
  create_graph(graph_id)

  _add_node(str(0),graph_id)

  for k,v in AllHosts.items():
    _add_node(str(k)+'_i',graph_id)
    _add_node(str(k)+'_f',graph_id)

  for k,v in MBList.items():
    _add_node(k,graph_id)

  _add_node(str(1),graph_id)

  mb1 = []
  mb2 = []
  mb3 = []

  for k,v in MBList.items():
    if v[2] == mbt1:
      mb1.append(k)
    elif v[2] == mbt2:
      mb2.append(k)
    elif v[2] == mbt3:
      mb3.append(k)

  for k,v in AllHosts.items():
    add_edge(str(k)+'_i',str(0),0,graph_id)
    add_edge(str(k)+'_f',str(1),0,graph_id)

  #print mb1
  for k,v in AllHosts.items():
    for mb1_ in mb1:
      #print k,mb1_,shortest_distance(k, mb1_, 0)
      add_edge(str(k)+'_i',mb1_,shortest_distance(k, mb1_, 0),graph_id)
  
  for mb1_ in mb1:
    for mb2_ in mb2:
      add_edge(mb1_,mb2_,shortest_distance(mb1_, mb2_, 0),graph_id)

  for mb2_ in mb2:
    for mb3_ in mb3:
      add_edge(mb2_,mb3_,shortest_distance(mb2_, mb3_, 0),graph_id)

  for mb3_ in mb3:
    for k,v in AllHosts.items():
      add_edge(mb3_,str(k)+'_f',shortest_distance(mb3_, k, 0),graph_id)

  
########################################


def getActiveVms():
    vmids = []
    for k,v in flows.items():
        vmids.append(v[1])
        vmids.append(v[2])

    vmids = list(set(vmids))
    return vmids

def getVmFlows(vm):
  k = []
  for i,v in flows.items():
      if (v[1] == vm or v[2] == vm) and str(i).startswith('b_') == False:
          k.append([i,v[1],v[2]])
  
  k.sort()
  VmFlows = list(k for k,_ in itertools.groupby(k))
  return VmFlows


def getNCVG():
  del Groups[:]

  vmids = getActiveVms()
  
  while len(vmids) > 0:
      Group = []
      Group.append(vmids[0])
      VmFlows = getVmFlows(vmids[0])
      while (len(VmFlows) > 0):
          fj = VmFlows[0]
          if fj[1] not in Group:
              vp = fj[1]
          elif fj[2] not in Group:
              vp = fj[2]
          else:
              VmFlows.remove(fj)
              continue
          Group.append(vp)
          k = VmFlows + getVmFlows(vp)
          k.sort()
          VmFlows = list(k for k,_ in itertools.groupby(k))
          VmFlows.remove(fj)
  
      Groups.append(Group)
      vmids = [vm for vm in vmids if vm not in Group]
  if len(Groups) > 0:
      print 'Printing Vm Groups...'
      for g in Groups:
          print g

def vmMigCost(vm, h):
  if vms[vm][1] != h:
    return 3
  else:
    return 0

def initialiseRo():
  global ro
  for k,v in AllHosts.items():
    for k1,v1 in vms.items():
      ro.append([k,k1,0])

def updateRo(s,v):
  for i in ro:
    if i[0] == s and i[1] == v:
      i[2] = int(i[2]) + 1
  

def SPF(f):
  p = flows[f][6]
  vm_s = flows[f][1]
  vm_d = flows[f][2]
  mblist = policies[p]
  mb1t = MBList[str(mblist[0])][2]
  mb2t = MBList[str(mblist[1])][2]
  mb3t = MBList[str(mblist[2])][2]
  #shortest_distance(g,'h1','mb1') 

  min_ = 1000

  for hs,vhs in AllHosts.items():
    for mb1,vmb1 in MBList.items():
      if vmb1[2] == mb1t:
        for mb2,vmb2 in MBList.items():
	  if vmb2[2] == mb2t:
	    for mb3,vmb3 in MBList.items():
	      if vmb3[2] == mb3t:
	        for hd,vhd in AllHosts.items():
		  d = vmMigCost(vm_s,hs)+shortest_distance(hs,mb1)+shortest_distance(mb1,mb2)+shortest_distance(mb2,mb3)+shortest_distance(mb3,hd)+vmMigCost(vm_d,hd)
		  
		  if d <= min_:
		    min_ = d
		    path = hs+'->'+mb1+'('+mb1t+')->'+mb2+'('+mb2t+')->'+mb3+'('+mb3t+')->'+hd+'='+str(d)
		    mlist = [mb1,mb2,mb3]

  #print 'Flow:', f, 'Policy:',p, path, hs,hd,mlist
  return hs,hd,mlist


def migratePolicy(f,mlist):
  p = flows[f][6]
  mblist0 = policies[p]
  f_c = flows[f][0]

  if f_c <= MBList[mlist[0]][4]:
    mb1 = mlist[0]
    if mb1 != mblist0[0]:
      MBList[mblist0[0]][4] = int(MBList[mblist0[0]][4]) - int(f_c)
  else:
    mb1 = mblist0[0]

  if f_c <= MBList[mlist[0]][4]:
    mb2 = mlist[1]
    if mb2 != mblist0[1]:
      MBList[mblist0[1]][4] = int(MBList[mblist0[1]][4]) - int(f_c)
  else:
    mb2 = mblist0[1]

  if f_c <= MBList[mlist[0]][4]:
    mb3 = mlist[2]
    if mb3 != mblist0[2]:
      MBList[mblist0[2]][4] = int(MBList[mblist0[2]][4]) - int(f_c)
  else:
    mb3 = mblist0[2]

  policies[p] = [mb1, mb2, mb3]
  policies['b_'+str(p)] = [mb3, mb2, mb1]
  
def getVmFlowsGroup(vm_group):
  vms_flows = []

  for vm in vm_group:
    vms_flows = vms_flows + getVmFlows(vm)
    
  vms_flows.sort()
  vms_flows = list(vms_flows for vms_flows,_ in itertools.groupby(vms_flows))
  
  return vms_flows

 
def phase_I(vm_group): 
  #initialiseRo()
  #getNCVG()
  #if len(Groups) > 0:
  vms_flows = getVmFlowsGroup(vm_group)
  print 'vms_flows:',vms_flows
  for f in vms_flows:
    fk = f[0]
    #print fk
    pk = flows[fk][6]
    #print pk
    src,dst,mlist = SPF(fk)
    #print src,dst,mlist
    print 'policy ', pk,' before migration:',policies[pk]
    migratePolicy(fk,mlist)
    print 'policy ', pk,' after migration:',policies[pk]
    vm_s = flows[fk][1]
    vm_d = flows[fk][2]
    updateRo(src,vm_s)
    updateRo(dst,vm_d)



def getVmFlows_cost(vm):
  k = []
  for i,v in flows.items():
      if (v[1] == vm or v[2] == vm) and str(i).startswith('b_') == False:
          k.append([i,v[0],v[1],v[2],v[6],v[10],v[11]])

  if len(k)>0:
    k.sort()
    VmFlows = list(k for k,_ in itertools.groupby(k))
  VmFlows = k
  return VmFlows
      

def calcTotCommCost(vm, s):
  vmFlows = getVmFlows_cost(vm) 
  #outgoing traffic from vm in s
  o = 0
  i = 0
  if len(vmFlows)>0:
    for f in vmFlows:
      if f[1] == vm:
        o = int(o) + int(f[1]) * int(shortest_distance(s,policies[f[4]][0]))
      elif f[2] == vm:
        i = int(i) + int(f[1]) * int(shortest_distance(s,policies[f[4]][2]))

  #print 'calcTotCommCost of:',vm, s,' is:', int(o)+int(i)
  return int(o)+int(i) 

def utilityOfMig(vm,s):
  vm_s = ''
  for k,v in vms.items():
    if k == vm:
      vm_s = v[1] 

  return calcTotCommCost(vm,vm_s) - calcTotCommCost(vm,s) - vmMigCost(vm,s)

def obtainPrefList():
  global prefList
  prefList = {}
  for s,v in AllHosts.items():
    l = []
    for item in ro:
      if item[0] == s:
	l.append([item[1],item[2]])
    l.sort(key=lambda x: x[1], reverse=True)
    #print l
    lv = []
    for i in l:
      lv.append(i[0])  
    prefList[s] = lv

def initialiseBList(vm_group):
  #global blackList
  blackList = {}
  for vm in vm_group:
    blackList[vm] = []

  return blackList

def getMaxUtility(vm, blacklist):
  #obtain S(v) \ blacklist
  vm_s = vms[vm][1]
  vm_c = vms[vm][2]
  list_srvs = []
  for k,v in AllHosts.items():
    if v[2] >= vm_c and k != vm_s and k not in blackList[vm]:
      list_srvs.append(k)

  #determine sj
  max_ = -1000
  sj = None
  #print 'list_srvs in getMaxUtility:',list_srvs
  for s in list_srvs:
    #print 'utilityOfMig(g,vm,s):',vm,s,utilityOfMig(g,vm,s)
    if float(utilityOfMig(vm,s)) >= float(max_):
      max_ = utilityOfMig(vm,s)
      sj = s

  #print 'sj:',sj
  return sj 
  

def checkSrvCapacity(sj, newhvm):
  #vm_s = vms[vm][1] #is always different to s
  #vm_c = vms[vm][2]
  if len(hostedVMs[sj]) > 0 and len(newhvm) > 0:
    l = list(set(newhvm + hostedVMs[sj]))
  elif len(newhvm) > 0:
    l = list(set(newhvm))
  else:
    l = []

  l1 = [x for x in l if x not in hostedVMs[sj]]
  vm_c = 0
  if len(l1) > 0:
    for i in l1:
      vm_c = vm_c + int(vms[i][2])

  s_s = AllHosts[sj][2]
  if s_s >= vm_c:
    return True
  else:
    return False

def getUnprocessedVms(newAlloc):
  l = []
  for k,v in newAlloc.items():
    if v == None:
      l.append(k)
  return l

def lastVmInPrefList(sj, newHostedVms):
  lj = prefList[sj]
  lj1 = [x for x in lj if x in newHostedVms]
  if len(lj1) > 0:
    return lj1[-1]
  else:
    return -1


def phase_II(vm_group):
  global newAlloc
  global exitAlloc
  global hostedVMs
  global blackList
  
  newAlloc = {}
  for v in vm_group:
    newAlloc[v] = None

  hostedVMs = {}
  for h,hv in AllHosts.items():
    l = []
    for k,v in vms.items():
      if v[1] == h:
        l.append(h)    
    hostedVMs[h] = l

  newHostedVms = defaultdict(list)

  obtainPrefList()
  blackList = initialiseBList(vm_group)

  while len(getUnprocessedVms(newAlloc)) > 0:
    l = getUnprocessedVms(newAlloc)
    vm = l[0]
    #print 'before getMaxUtility:',vm, blackList[vm]
    sj = getMaxUtility(vm,blackList[vm])
    newAlloc[vm] = sj
    #vm_on_s_g: group of all vm on server sj before and after migration
    newHostedVms[sj].append(vm)
    #print 'len of newHostedVms[sj] before checkSrvCapacity:', sj, len(newHostedVms[sj])
    if checkSrvCapacity(sj, newHostedVms[sj]) == False:
      while True:
	vk = lastVmInPrefList(sj, newHostedVms[sj])
	newAlloc[vk] = None
	newHostedVms[sj].remove(vk)
	best_rejected = vk
	if checkSrvCapacity(sj, newHostedVms[sj]) == True:
	  break
      #print 'best_rejected:',best_rejected        
      #print 'blackList:',blackList
      #print 'prefList[sj]:',prefList[sj]
      for vk_ in [x for x in prefList[sj] if x in vm_group]:
	#print 'vk_:',vk_
        if vk_ != best_rejected:
	  blackList[vk_].append(sj)
	elif vk_ == best_rejected:
	  blackList[vk_].append(sj)
	  break
      #print 'size blackList:', len(blackList)

  existAlloc = {}
  for k,v in vms.items():
    if k in vm_group:
      existAlloc[k] = v[1]
  print 'Exisiting Allocation:', existAlloc
  print 'New Allocation:', newAlloc
  #for vm,s in newAlloc.items():
  #  migrateVM(net, vm, s)

 
class MyTopo(Topo):
    "Simple loop topology example."
 
    def __init__(self):
        "Create 4-k fattree topo."
 
        # Initialize topology
        Topo.__init__(self)
	k = 14
	#k=6 for 1k and 2k vms
	#k=8 for 2k, 3k, and 5k vms
        self.pod = k
	self.end = self.pod/2
	self.iCoreLayerSwitch = (k/2)**2
	self.iAggLayerSwitch = k*(k/2)
	self.iEdgeLayerSwitch = k*(k/2)
	self.iHost = self.iEdgeLayerSwitch * (k/2)
	self.SCount = 0
	print 'Number of edge switches:',self.iEdgeLayerSwitch
	print 'Number of agg switches:',self.iAggLayerSwitch
	print 'Number of core switches:',self.iCoreLayerSwitch
	print 'Number of switches:',int(self.iEdgeLayerSwitch)+int(self.iAggLayerSwitch)+int(self.iCoreLayerSwitch)
	print 'Number of hosts:',self.iHost
        get_ack_data_thread = threading.Thread(target=get_ack_data)
        get_ack_data_thread.daemon = True
        get_ack_data_thread.start()

        send_data_10s_thread = threading.Thread(target=send_data_10s)
	send_data_10s_thread.daemon = True
        send_data_10s_thread.start()

	get_sync_data_socket_thread = threading.Thread(target=get_sync_data_socket)
	get_sync_data_socket_thread.daemon = True
	get_sync_data_socket_thread.start()

	migrateVMs_thread = threading.Thread(target=migrateVMs)
	migrateVMs_thread.daemon = True
        migrateVMs_thread.start()

	get_msg_socket_thread = threading.Thread(target=get_msg_socket)
	get_msg_socket_thread.daemon = True
	get_msg_socket_thread.start()
	
	if not os.path.exists("/tmp/sync/"):
            os.makedirs("/tmp/sync/")

	global nvm
	global np
	global nf 
	global server_c	
	global mb_capacity
	global mb_type
	global fmin
	global fmax
	global nb_mbs
	global nb_policy

	create_graph(0)#for topology
	#print graphs[0]

	nvm = 100000#2000000#5000#10000#100#70 for test
	nb_mbs = 50 #k = 14 hosts=686, agg switches=98
	nb_policy = nb_mbs*10
	np = (self.iAggLayerSwitch-3)//3#10#50 #depends on the number of mbs (self.iAggLayerSwitch-2)
	nf = 100000##3000#50#7 for test

	vm_c = [0.0001,0.0002,0.0003]
	mb_type = ['FW', 'IPS', 'Proxy']
	mb_capacity = [100000, 200000, 300000, 400000, 500000, 600000]#kb/s
	server_c = 20#inf # at most 20 vms per host 
	fmin = 100#kbit/s
	fmax = 200
	
	#linkopts_host_edge = dict(bw=5, delay='5ms', loss=10, max_queue_size=10, use_htb=True)
	self.linkopts_host_edge = dict(bw=1)#Mbit/s
	#linkopts_edge_aggr = dict(bw=10, delay='5ms', loss=10, max_queue_size=100, use_htb=True)
	self.linkopts_edge_aggr = dict(bw=5)
	#linkopts_aggr_core = dict(bw=100, delay='5ms', loss=10, max_queue_size=1000, use_htb=True)
	self.linkopts_aggr_core = dict(bw=10)
	

    def createSwitches(self):
	#for x in range(1, self.pod*(self.pod/2)+1):
	#    PREFIX = "s"
    	    #EdgeSwitchList.append(self.addSwitch(PREFIX + str(x), protocols='OpenFlow13'))
	#    EdgeSwitchList.append(PREFIX + str(x))	    
    	#    self.SCount = self.SCount+1
     	    #print "ESwitch[",self.SCount,"]"
	#    _add_node(PREFIX + str(x),0)

	#for x in range(self.SCount+1,self.SCount+self.pod*(self.pod/2)+1):
	#  PREFIX = "s"
	  #AggSwitchList.append(self.addSwitch(PREFIX + str(x), protocols='OpenFlow13'))
	#  AggSwitchList.append(PREFIX + str(x))
	#  self.SCount = self.SCount+1
	  #print "ASwitch[",self.SCount,"]"
	#  _add_node(PREFIX + str(x),0)

	#for x in range(self.SCount+1,self.SCount+((self.pod/2)**2)+1):
	for x in range(0,((self.pod/2)**2)):
	  PREFIX = "s"
	  #CoreSwitchList.append(self.addSwitch(PREFIX + str(x), protocols='OpenFlow13'))
	  CoreSwitchList.append(PREFIX + str(x))
	  self.SCount = self.SCount+1
	  #print "CSwitch[",self.SCount,"]"
	  _add_node(PREFIX + str(x),0)

        for x in range(self.SCount,self.SCount+self.pod*(self.pod/2)):
          PREFIX = "s"
          #AggSwitchList.append(self.addSwitch(PREFIX + str(x), protocols='OpenFlow13'))
          AggSwitchList.append(PREFIX + str(x))
          self.SCount = self.SCount+1
          #print "ASwitch[",self.SCount,"]"
          _add_node(PREFIX + str(x),0)

        for x in range(self.SCount, self.SCount+self.pod*(self.pod/2)):
            PREFIX = "s"
            #EdgeSwitchList.append(self.addSwitch(PREFIX + str(x), protocols='OpenFlow13'))
            EdgeSwitchList.append(PREFIX + str(x))
            self.SCount = self.SCount+1
            #print "ESwitch[",self.SCount,"]"
            _add_node(PREFIX + str(x),0)



    def createHosts(self):
	#f_hosts = open('/tmp/sync/hosts.csv', 'w')
	e = 0
	f = 0
	count = 0
	digit2 = 0
	digit3 = 0
	for a in range(0,self.pod):
	  for b in range(0,self.pod/2):
	    for c in range(2,2+(self.pod/2)+1):#+1 for masse hosts
	      if c != 2+(self.pod/2): 
		
		s = ((self.pod/2)+f) // (self.pod/2)
		f =  f + 1
		#print 's',s
                count = count+1
                digit2 = count/100
                digit3 = count/10000
                PREFIX = "h"
	        #print "digit2:",digit2
	        #print "digit3:",digit3
	        #print "count:",count
	        #print "host ip:","10."+str(a)+"."+str(b)+"."+str(c)
	        #print "host mac:","00:00:00:"+str(digit3%100).zfill(2)+":"+str(digit2%100).zfill(2)+":"+str(count%100).zfill(2)
	        #f1.write(PREFIX + str(count) + " " + "00:00:00:"+str(digit3%100).zfill(2)+":"+str(digit2%100).zfill(2)+":"+str(count%100).zfill(2)+"\n")
	        #HostList.append(self.addHost(PREFIX + str(count),ip="10."+str(a)+"."+str(b)+"."+str(c),mac="00:00:00:"+str(digit3%100).zfill(2)+":"+str(digit2%100).zfill(2)+":"+str(count%100).zfill(2)))#, cpu=.5/iHost
		HostList.append(PREFIX + str(self.SCount+count-1))
	        AllHosts[PREFIX + str(self.SCount+count-1)] = ["10."+str(a)+"."+str(b)+"."+str(c), "00:00:00:"+str(digit3%100).zfill(2)+":"+str(digit2%100).zfill(2)+":"+str(count%100).zfill(2), server_c, 's'+str(s)]
	        #h_id = PREFIX + str(count)
	        #print h_id
	        #f_hosts.write(str(h_id)+","+str(AllHosts[h_id][0])+","+str(AllHosts[h_id][1])+','+str(AllHosts[h_id][2])+"\n")
	        _add_node(PREFIX + str(self.SCount+count-1),0)
	      else:
		e = e+1
		e2 = e/100
		e3= e/10000
		#print "host ip:","10."+str(a)+"."+str(b)+"."+str(c)
		#print "host mac:","00:00:00:00:02:"+str(e).zfill(2)
		#MassesList.append(self.addHost('m' + str(e),ip="10."+str(a)+"."+str(b)+"."+str(c),mac="00:00:01:"+str(e3%100).zfill(2)+":"+str(e2%100).zfill(2)+":"+str(e%100).zfill(2)))#, cpu=.5/iHost
		MassesList.append('m' + str(e))
                masses['m' + str(e)] = ["10."+str(a)+"."+str(b)+"."+str(c), "00:00:01:"+str(e3%100).zfill(2)+":"+str(e2%100).zfill(2)+":"+str(e%100).zfill(2),'s'+str(e)]
		#print 'masses:', masses['m' + str(e)]
	#f_hosts.close()
	#f2=open('/tmp/sync/f2.csv', 'w')

    def createLinks(self):
	e = 0
	for x in range(0, self.iEdgeLayerSwitch):
	  for y in range(0,self.end+1):#+1 for masse hosts
	    if y != self.end:
	      #self.addLink(EdgeSwitchList[x], HostList[self.end*x+y], **self.linkopts_host_edge)
	      #print 'link between:',EdgeSwitchList[x], HostList[self.end*x+y]
	      add_edge(str(EdgeSwitchList[x]), str(HostList[self.end*x+y]), 1,0)
	    else:
	      #self.addLink(EdgeSwitchList[x], MassesList[e], **self.linkopts_host_edge)
              #print 'link between:',EdgeSwitchList[x], MassesList[e]
              add_edge(str(EdgeSwitchList[x]), str(MassesList[e]), 1, 0)
	      e = e + 1
	    #f2.write(str(HostList[end*x+y]) + " " + str(EdgeSwitchList[x])[1] + " " + str(y+1) +"\n")
	#f2.close()

	#print "iAggLayerSwitch=",self.iAggLayerSwitch
	for x in range(0, self.iAggLayerSwitch):
	  for y in range(0,self.end):
	   #self.addLink(AggSwitchList[x], EdgeSwitchList[self.end*(x/self.end)+y], **self.linkopts_edge_aggr)
	   add_edge(str(AggSwitchList[x]), str(EdgeSwitchList[self.end*(x/self.end)+y]), 3,0)

	for x in range(0, self.iAggLayerSwitch, self.end):
	  for y in range(0,self.end):
	    for z in range(0,self.end):
	      #self.addLink(CoreSwitchList[y*self.end+z], AggSwitchList[x+y], **self.linkopts_aggr_core)
	      add_edge(str(CoreSwitchList[y*self.end+z]), str(AggSwitchList[x+y]), 5,0)

	# add mbs
    def createMBs(self):
	global AllSwitches
	AllSwitches = AggSwitchList #EdgeSwitchList + AggSwitchList + CoreSwitchList
	SwitchIndex = []
	#print len(AllSwitches)
	for i in range(0, len(AllSwitches)):
	    SwitchIndex.append(i)
	
	#for h in AllHosts:
	#    print h;
	#MBList.append(self.addHost('mbtest'))
	#self.addLink(AggSwitchList[0], MBList[0])
	#MBList.append(self.addHost("mb1"))
	#mb1 = self.addHost("mb1")
	#mb1.cmd("ifconfig mb1-eth0 192.168.1.1 netmask 255.255.255.0")
	#mb1.cmd("ifconfig mb1-eth1 192.168.1.2 netmask 255.255.255.0")
	#mb1.cmd("echo 1 > /proc/sys/net/ipv4/ip_forward")
	#f_mb=open('/tmp/sync/mbs.csv', 'w')

	#mb_capacity = [50000, 60000, 70000, 80000, 90000, 100000]#kb/s
	#mb_type = ['FW', 'IPS', 'Proxy']
	c1 = 0
	#print SwitchIndex
	for i in range(1, nb_mbs+1):#max len(AllSwitches)+1
	    c1 = c1+1
	    c2 = c1/100
	    c3 = c1/10000
            a1 = 255
            a2 = i//255
            a3 = i%255
            if a3 != 0:
	      t = random.choice(mb_type)
	      c = random.choice(mb_capacity)#Kbit/s
	      #mb = self.addHost("mb"+str(i), ip="10."+str(a1)+"."+str(a2)+"."+str(a3), mac="02:00:00:"+str(c3%100).zfill(2)+":"+str(c2%100).zfill(2)+":"+str(c1%100).zfill(2))
	      mb = "mb"+str(i)
	      s = random.choice(SwitchIndex)
	      SwitchIndex.remove(s)#no two mbs on the same switch
	      #self.addLink(mb, AllSwitches[s])
	      #print 'info:', info
	      _add_node(str(mb),0)
	      add_edge(str(mb), str(AllSwitches[s]), 0,0)
	      #max number of pod = 253 = k too big..
	      MBList[mb] = ["10."+str(a1)+"."+str(a2)+"."+str(a3), "02:00:00:"+str(c3%100).zfill(2)+":"+str(c2%100).zfill(2)+":"+str(c1%100).zfill(2), t, AllSwitches[s], c]
	    print MBList[mb],'associated to',str(AllSwitches[s])
	    #f_mb.write(str(mb)+","+str(MBList[mb][0])+"\n")
	    #f_mb.write(str(mb)+","+str(MBList[mb][0])+","+str(MBList[mb][1])+","+str(MBList[mb][2])+","+str(MBList[mb][3])+','+str(MBList[mb][4])+"\n")
	#f_mb.close()


#dump_data

def dump_data(runnum):
  f_mb=open("/tmp/sync/mbs_"+str(runnum)+".csv", 'w')
  for k,v in MBList.items():
    f_mb.write(str(k)+","+str(v[0])+","+str(v[1])+","+str(v[2])+","+str(v[3])+','+str(v[4])+"\n")
  f_mb.close()
  
  f_hosts = open('/tmp/sync/hosts_'+str(runnum)+'.csv', 'w')
  for k,v in AllHosts.items():
    f_hosts.write(str(k)+","+str(v[0])+","+str(v[1])+','+str(v[2])+','+str(v[3])+"\n")
  f_hosts.close()

  f_f = open('/tmp/sync/flows_'+str(runnum)+'.csv', 'w')
  for k,v in flows.items():
    f_f.write(str(k)+','+str(v[0])+','+str(v[1])+','+str(v[2])+','+str(v[3])+','+str(v[4])+','+str(v[5])+','+str(v[6])+','+str(v[7])+','+str(v[8])+','+str(v[9])+','+str(v[10])+','+str(v[11])+'\n')
  f_f.close()

  f_p = open('/tmp/sync/policies_'+str(runnum)+'.csv', 'w')
  for k,v in policies.items():
    f_p.write(str(k)+","+str(v[0])+","+str(v[1])+","+str(v[2])+"\n")
  f_p.close()

  f_v = open('/tmp/sync/vms_'+str(runnum)+'.csv', 'w')
  for k,v in vms.items():
    f_v.write(str(k)+','+str(v[0])+','+str(v[1])+','+str(v[2])+','+str(v[3])+'\n')
  f_v.close()

  f_m = open('/tmp/sync/masses_'+str(runnum)+'.csv', 'w')
  for k,v in masses.items():
    f_m.write(str(k)+','+str(v[0])+','+str(v[1])+','+str(v[2])+'\n')
  f_m.close()

  #f_pi = open('/home/mininet/ryu/ryu/app/sync/pingmb.sh', 'w')
  #for k,v in MBList.items():
  #  f_pi.write('ping -c1 '+str(v[0])+'\n')
  #f_pi.close()



def get_vm_ip(vm):
  h = None
  for k,v in vms.items():
    if str(k) == str(vm):
      h = v[1]
      #print k,vm,h
      #break
  
  for k,v in AllHosts.items():
    if str(k) == str(h):
      #print k,h,v[0]
      return v[0]
  
def get_masse_ip(vm):
  h = ''
  for k,v in vms.items():
    if k == vm:
      h = v[1]
  for k,v in AllHosts.items():
    if k == h:
      s = v[3]  

  for k,v in masses.items():
    if v[2] == s:
      return v[0]
      

#create flows
# if src == dst --> masse idea... 
def createFlows():
    vmids = []
    pcids = []
    protos = ['tcp']#'icmp',udp not included

    #for k,v in vms.items():
    #    vmids.append(k)
    for k,v in policies.items():
	if str(k).startswith('b_') == False:
          pcids.append(k)

    s_port = 1001
    d_port = 2001
    fid = 1
    i = 0
    j = 0
    
    while i < nf:
	#if i != a:
	#if a > len(vms)*4:
	#  break
	j = j + 1
	if j > nf*2:
	    break
	#print len(vms)*4,a
	policy = random.choice(pcids)
	#proto = random.choice(protos)
	proto = 'tcp'
	#vmids_ = vmids[:]
	#vm1 = random.choice(vmids_)
	#vmids_.remove(vm1)
	#vm2 = random.choice(vmids_)
	v = random.randint(1,len(vms)-1)
	vm1 = "vm"+str(v)
	
	vm2 = "vm"+str(v+1) 
	
	flowid = "f" + str(fid) 
	#flowrate = random.randint(fmin, fmax)#Kbit/s
	flowrate=1
	issameip = 0
	if vms[vm1][3] == vms[vm2][3]:
	    issameip = 1
	if createFlow(flowid, flowrate, vm1, vm2, s_port, d_port, proto, policy, issameip) == 0:
	    print 'created flow:',flowid, ' between:',vm1,vm2, ' with policy:',policy
	    i = i + 1
	    fid = fid + 1
	    if proto != 'icmp':
		s_port = s_port + 1
	        d_port = d_port + 1
    #f_f.close()

#start only original flows !! starts with f

def get_host(ip):
  h = []
  for k,v in AllHosts.items():
    h.append([k,v[0]])

  for k,v in masses.items():
    h.append([k,v[0]])

  for i in h:
    if i[1] == ip:
      return i[0]
 
'''
def startFlow(net, flowid):
  #if str(flowid).startswith('b_') == False: 
    flowrate = flows[flowid][0]
    #flowrate_inm = flowrate * 0.001
    vm1 = flows[flowid][1]
    vm2 = flows[flowid][2]
    s_port = flows[flowid][3]
    d_port = flows[flowid][4]
    proto = flows[flowid][5]
    #host1 = vms[vm1][1]
    #host2 = vms[vm2][1]
    issameip = flows[flowid][7]
    vm1_ip = flows[flowid][8]
    vm2_ip = flows[flowid][9]

    host1 = get_host(vm1_ip)
    host2 = get_host(vm2_ip)

    client = net.get(host1)
    server = net.get(host2)

    if proto == 'tcp':
	server.cmd('>server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        server.cmd('iperf3 -s -p ' + str(d_port) + ' >> server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
        srv_pid = server.cmd('echo $!')
        waitListening( client, server, d_port )
	client.cmd('>client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        client.cmd('iperf3 -c ' + server.IP() + ' -B ' + client.IP() + ' --cport ' + str(s_port) + ' -p ' + str(d_port) + ' -b '+ str(flowrate) + 'k -t 1' + ' >> client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
	clt_pid = client.cmd('echo $!')
	flows[flowid][12] = clt_pid
	flows[flowid][13] = srv_pid
    elif proto == 'udp':
	server.cmd('>server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        server.cmd('iperf3 -s -p ' + str(d_port) + ' >> server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
        waitListening( client, server, d_port )
	client.cmd('>client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        client.cmd('iperf3 -c ' + server.IP() + ' -B ' + client.IP() + ' --cport ' + str(s_port) + ' -u -p ' + str(d_port) + ' -b '+ str(flowrate) + 'k -t 1' + ' >> client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
    elif proto == 'icmp':
	client.cmd('>client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
	client.cmd('ping -c10 ' + server.IP() + ' >> client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
	clt_pid = client.cmd('echo $!')
	flows[flowid][12] = clt_pid
'''

def startFlow(net, flowid):
  if str(flowid).startswith('b_') == False:
    flowrate = flows[flowid][0]
    #flowrate_inm = flowrate * 0.001
    vm1 = flows[flowid][1]
    vm2 = flows[flowid][2]
    s_port = flows[flowid][3]
    d_port = flows[flowid][4]
    proto = flows[flowid][5]
    #host1 = vms[vm1][1]
    #host2 = vms[vm2][1]
    issameip = flows[flowid][7]
    vm1_ip = flows[flowid][8]
    vm2_ip = flows[flowid][9]

    host1 = get_host(vm1_ip)
    host2 = get_host(vm2_ip)

    client = net.get(host1)
    server = net.get(host2)
    buff_len = 1458#bytes
    p = flows[flowid][6]
    print 'start flow:',flowid,' from:', vm1,host1,vm1_ip, ' to:',vm2,host2,vm2_ip, ' through:',p,' :',policies[p][0],policies[p][1],policies[p][2]
    if proto == 'tcp':
        #server.cmdPrint('>server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        server.cmdPrint('iperf3 -s -D -p ' + str(d_port))# + ' &')#>> server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
        #srv_pid = server.cmdPrint('echo $!')
        #waitListening( client, server, d_port )
        #client.cmdPrint('>client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        client.cmdPrint('iperf3 -c ' + server.IP() + ' -B ' + client.IP() + ' --cport ' + str(s_port) + ' -p ' + str(d_port) + ' -b '+ str(flowrate) + 'k -l 1458 -t 1')# + ' >> client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
        #clt_pid = client.cmdPrint('echo $!')
        #flows[flowid][12] = clt_pid
        #flows[flowid][13] = srv_pid
    elif proto == 'udp':
        #server.cmdPrint('>server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        server.cmdPrint('iperf3 -s -D -p ' + str(d_port))# + ' &')#>> server_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
        #waitListening( client, server, d_port )
        #client.cmdPrint('>client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        client.cmdPrint('iperf3 -c ' + server.IP() + ' -B ' + client.IP() + ' --cport ' + str(s_port) + ' -u -p ' + str(d_port) + ' -b '+ str(flowrate) + 'k -l 1458 -t 1')# + ' >> client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
    elif proto == 'icmp':
        #client.cmdPrint('>client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log')
        client.cmdPrint('ping -c10 ' + server.IP())# + ' >> client_'+str(flowid)+'_'+vm1+'_'+vm2+'.log &')
        #clt_pid = client.cmdPrint('echo $!')
        #flows[flowid][12] = clt_pid

def stopFlow(net, flowid):
  if str(flowid).startswith('b_') == False:
    proto = flows[flowid][5]
    issameip = flows[flowid][7]    

    vm1_ip = flows[flowid][8]
    vm2_ip = flows[flowid][9]

    host1 = get_host(vm1_ip)
    host2 = get_host(vm2_ip)

    client = net.get(host1)
    server = net.get(host2)

    clt_pid = flows[flowid][12]

    if proto == 'tcp' or proto == 'udp':
      srv_pid = flows[flowid][13]
      server.cmdPrint('kill -9 ' + str(srv_pid))
      client.cmdPrint('kill -9 ' + str(clt_pid))
    else:
      client.cmdPrint('kill -9 ' + str(clt_pid))
      

def createFlow(flowid, flowrate, vm1, vm2, s_port, d_port, proto, policy, issameip):
    #flowrate between 1000Kbit and 5000Kbit, 1Kbit = ^-3 Mbit
    mb_list = policies[policy]
    mb1 = mb_list[0]
    mb2 = mb_list[1]
    mb3 = mb_list[2]
    mb1_c = MBList[mb1][4]
    mb2_c = MBList[mb2][4]
    mb3_c = MBList[mb3][4]

    vm1_ip = vms[vm1][3]
    host1 = vms[vm1][1]
    '''
    if issameip == 1:
      #print '>>>>>>>>>>>>>>>>>> issameip', issameip
      #vm2_ip_ = vms[vm2][3]
      vm2_ip = get_masse_ip(vm2)
      for k,v in masses.items():
	if v[0]==vm2_ip:
	  host2 = k
	  break
      #host2 = masses[
      #host2 = vms[vm2][1]
    else:
    '''
    vm2_ip = vms[vm2][3]
    host2 = vms[vm2][1]

    
    #host1 = get_host(vm1_ip)
    #host2 = get_host(vm2_ip)

    if flowrate <= mb1_c and flowrate <= mb2_c and flowrate <= mb3_c:
	flows[flowid] = [flowrate, vm1, vm2, s_port, d_port, proto, policy, issameip, vm1_ip, vm2_ip, host1, host2,-1,-1]
	flows['b_'+str(flowid)] = [flowrate, vm2, vm1, d_port, s_port, proto, 'b_'+str(policy), issameip, vm2_ip, vm1_ip, host2, host1,-1,-1]#used only for routing
	MBList[mb1][4] = MBList[mb1][4] - flowrate
	MBList[mb2][4] = MBList[mb2][4] - flowrate
	MBList[mb3][4] = MBList[mb3][4] - flowrate
	return 0
    else: 
	return -1

def updateFlow(flowid,vm1,vm2):#called after vm migration
  vm1_ip = vms[vm1][3]
  vm2_ip = vms[vm2][3]
  issameip = None
  host1 = None
  host2 = None
  if str(vm1_ip) == str(vm2_ip):
    issameip = 1
    vm2_ip = get_masse_ip(vm2)
    host1 = get_host(vm1_ip)
    host2 = get_host(vm2_ip)
    flows[flowid][7] = issameip
    flows[flowid][8] = vm1_ip
    flows[flowid][9] = vm2_ip
    flows[flowid][10] = host1
    flows[flowid][11] = host2
  else:
    issameip = 0
    host1 = get_host(vm1_ip)
    host2 = get_host(vm2_ip)
    flows[flowid][7] = issameip
    flows[flowid][8] = vm1_ip
    flows[flowid][9] = vm2_ip
    flows[flowid][10] = host1
    flows[flowid][11] = host2

  flows['b_'+str(flowid)][7] = issameip
  flows['b_'+str(flowid)][8] = vm2_ip
  flows['b_'+str(flowid)][9] = vm1_ip
  flows['b_'+str(flowid)][10] = host2
  flows['b_'+str(flowid)][11] = host1

#create policies
def createPolicies_():
    vmids = []
    mbs = []
    mb1 = None
    mb2 = None
    mb3 = None
    for k,v in vms.items():
        vmids.append(k)
    for k,v in MBList.items():
        mbs.append(k)
    #print mbs
    #create policies of 3 mbs
    #f_p = open('/tmp/sync/policies.csv', 'w')
    #mb_type = ['FW', 'IPS', 'Proxy']
    a = 0
    b = 0
    i = 0
    #np = 70#len(vms)//2
    while i < np:
        if i > 2*np:
	  break
	exist = False
    #for i in range(0, len(vms)//2):
        mbt = mbs[:]
	#print 'size of mbt:', len(mbt)
	mbts = mb_type[:]
        #vm1 = random.choice(vmids)
        #vmids.remove(vm1)
        #vm2 = random.choice(vmids)
        #vmids.remove(vm2)

        mb1 = random.choice(mbt)
	mbts.remove(MBList[mb1][2])
	#print 'mb1',mbts
        mbt.remove(mb1)
	while True:
	  a = a + 1
          mb2 = random.choice(mbt)
	  if MBList[mb2][2] in mbts:
            mbt.remove(mb2)
	    mbts.remove(MBList[mb2][2])
	    #print 'mb2',mbts
	    break
	  if a > np:
	    break
	while True:
	  b = b + 1
          mb3 = random.choice(mbt)
          if MBList[mb3][2] in mbts:
	    mbt.remove(mb3)
	    mbts.remove(MBList[mb3][2])
	    #print 'mb3',mbts
	    break
	  if b > np:
	    break
	if mb1 == None or mb2 == None or mb3 == None:
	  continue
	elif MBList[mb1][2] == MBList[mb2][2] or MBList[mb1][2] == MBList[mb3][2] or MBList[mb2][2] == MBList[mb3][2]:
	  continue
	else:
	  for k,v in policies.items():
	    #print k, len(policies)
	    if str(k).startswith('b_') == False:
	      #print k, len(policies)
	      if v == [mb1,mb2,mb3]:
	        exist = True
	        break
	  if exist==False:
	    i = i + 1
            policies["p"+str(i)] = [mb1,mb2,mb3]
	    print "p"+str(i),mb1,mb2,mb3,MBList[mb1][2],MBList[mb2][2],MBList[mb3][2]
	    policies["b_p"+str(i)] = [mb3,mb2,mb1]
	  else:
	    continue
        #p_id = "p"+str(i+1)
        #f_p.write(str(p_id)+","+str(policies[p_id][0])+","+str(policies[p_id][1])+","+str(policies[p_id][2])+"\n")
    #f_p.close()
    #print policies

def createPolicies():
    mbs_type1 = []
    mbs_type2 = []
    mbs_type3 = []
    mbts = mb_type[:]

    for k,v in MBList.items():
      if v[2] == mbts[0]:
	mbs_type1.append(k)
    for k,v in MBList.items():
      if v[2] == mbts[1]:
	mbs_type2.append(k)
    for k,v in MBList.items():
      if v[2] == mbts[2]:
        mbs_type3.append(k)

    #print mbs_type1
    #print mbs_type2
    #print mbs_type3

    i = 1
    for mb1 in mbs_type1:
      for mb2 in mbs_type2:
        for mb3 in mbs_type3:
          policies["p"+str(i)] = [mb1, mb2, mb3]
	  policies["b_p"+str(i)] = [mb3, mb2, mb1]
	  i=i+1
	  if i>nb_policy:
	    return 0
	  policies["p"+str(i)] = [mb1, mb3, mb2]
	  policies["b_p"+str(i)] = [mb2, mb3, mb1]
	  i=i+1
          if i>nb_policy:
            return 0
	  policies["p"+str(i)] = [mb2, mb1, mb3]
	  policies["b_p"+str(i)] = [mb3, mb1, mb2]
	  i=i+1
          if i>nb_policy:
            return 0
	  policies["p"+str(i)] = [mb2, mb3, mb1]
	  policies["b_p"+str(i)] = [mb1, mb3, mb2]
	  i=i+1
          if i>nb_policy:
            return 0
	  policies["p"+str(i)] = [mb3, mb1, mb2]
	  policies["b_p"+str(i)] = [mb2, mb1, mb3]
	  i=i+1
          if i>nb_policy:
            return 0
	  policies["p"+str(i)] = [mb3, mb2, mb1]
	  policies["b_p"+str(i)] = [mb1, mb2, mb3]
      
    print 'number of generated policies is:',i


def testRun(net):
    client = net.get('h1')  
    server = net.get('h2')
    server.cmdPrint('iperf3 -s -p 2001 &')
    waitListening( client, server, 2001 )
    #print 'iperf3 -c ' + h2.IP() + ' -B ' + h1.IP() + ' --cport 1001 -p 2001 -b 1m -t 1'
    client.cmdPrint('iperf3 -c ' + server.IP() + ' -B ' + client.IP() + ' --cport 1001 -p 2001 -b 1m -t 1')


def createVMs(net):
    hosts = []
    for k,v in AllHosts.items():
        hosts.append(k)
    #j = 1
    #f_v = open('/tmp/sync/vms.csv', 'w')
    #f_v.write(str(vmname)+','+str(vms[str(vmname)][0])+','+str(vms[str(vmname)][1])+','+str(vms[str(vmname)][2])+'\n')
    #f_v.close()
    vm_c = [0.1,0.2,0.3]
    #nb of vms = len(AllHosts)*10
    #for i in range(0, len(AllHosts)*10):
    j = 1
    i = 0
    #nvm = 500#len(AllHosts)
    while i < nvm:#len(AllHosts):
	i = i + 1
	if i > 2*nvm:
	    break  
	server = random.choice(hosts)
	c = random.choice(vm_c)
	vmname = "vm" + str(j)
	#createVM(net, server, vmname, c)
	if createVM(net, server, vmname, c) == 0:
	    #f_v.write(str(vmname)+','+str(vms[str(vmname)][0])+','+str(vms[str(vmname)][1])+','+str(vms[str(vmname)][2])+'\n')
	    j = j + 1    
    #f_v.close()
    
def createVM(net, server, vmname, capacity):
    if float(AllHosts[str(server)][2]) >= float(capacity):
	#serverh = net.get(str(server))
	#serverh.cmd('while true; do sleep 1; done &')
	#pid = serverh.cmd('echo $!')
	#pid = pid.replace('\n', '').replace('\r', '')
        pid = 0
	vms[str(vmname)] = [pid, str(server), capacity, AllHosts[str(server)][0]]
	AllHosts[str(server)][2] = float(AllHosts[str(server)][2]) - float(vms[vmname][2])
        print 'created vm:',vmname, ' on:',server
	return 0
    else:
	return -1
    #print vmname," pid:", pid," on host:", server

def showFlows():
  for k,v in flows.items():
    print k,v

def migrateVM(net,vm,server):
  #server = net.get(str(server)) 
  vm_server = net.get(str(vms[vm][1]))
  #serverh = net.get(str(server))
  print 'Migration of vm:',vm,' from:',str(vms[vm][1]),' to:',str(server)
  #print showFlows()
  if str(vms[vm][1]) == str(server):
    print 'No migration needed'
    return 0
  else:
    if float(AllHosts[str(server)][2]) >= float(vms[vm][2]):
      print 'stop vm:',vm,' in:',str(vms[vm][1])
      #vm_server.cmd('kill -9 ' + str(vms[vm][0])) #stop and delete vm
      AllHosts[str(vm_server)][2] = float(AllHosts[str(vm_server)][2]) + float(vms[vm][2])
      createVM(net, server, vm, vms[vm][2]) #true
      #update routing for related flows
      VmFlows = getVmFlows(vm) #flowid,vm1,vm2
      if len(VmFlows) > 0:
        for f in VmFlows:
          flowid = f[0]
          vm1 = f[1]
          vm2 = f[2]
	  updateFlow(flowid,vm1,vm2)
        #print 'After vm migration:'
	print 'Migration done'
	#print showFlows()
	return 0
      else:
	print 'No flows for migrated vm:',vm,' in:',vms[vm][1]
        return 0
    else:
      print 'VM cannot be migrated because of insufficient server capacity',server,vm,float(AllHosts[str(server)][2]),float(vms[vm][2])
      return -1
        
      
 
def deleteVM(net, vmname):
    server = net.get(str(vms[vmname][1]))
    pid = vms[vmname][0]
    kill = "kill -9" + pid
    server.cmd(str(kill))
    del vms[str(vmname)]
    #print "kill vm:", vmname, " pid:",pid

def initialise(net):
    h1 = net.get('h1')
    for k,v in MBList.items():
        c = "ping -c1 " + k
	#print c
        h1.cmdPrint(c)

def send_rSync_socket(rSync):
  #rSync = True
  data = pickle.dumps(rSync)

  HOST = 'pi-head'
  PORT = 50011
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect((HOST, PORT))
    s.send(data)
    #print 'data sent to controller..'
  except socket.error, v:
    errorcode=v[0]
  s.close()
  #except socket.error, v:
  #  errorcode=v[0]
    #if errorcode==errno.ECONNREFUSED:
    #    print "Connection Refused"


def send_data_socket():#runSync boolean
  #print rSync
  global send_data_done
  #send_data_done = False
  l = []
  l = [AllHosts, MBList, policies, masses, CoreSwitchList, AggSwitchList, EdgeSwitchList, HostList]#, vms, flows]
  #l = [AllHosts]
  data = pickle.dumps(l)
  #data = str(l)
  global ack_data

  HOST = 'pi-head'
  PORT = 50007
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    ack_data = False
    send_data_done = False
    s.connect((HOST, PORT))
    send_msg(s, data)
    #s.send(data)
    #send_data_done = True
    #print 'send_data_socket - data sent to controller..' 
  except socket.error, v:
    errorcode=v[0]
  s.close()
  #except socket.error, v:
  #  errorcode=v[0]
    #if errorcode==errno.ECONNREFUSED:
    #    print "Connection Refused"

def send_topo_socket():
  #print len(graphs)
  l = [graphs,mymac_mb] #[nodes,edges,distances,_distance,mymac_mb]
  #l = [AllHosts]
  data = pickle.dumps(l)

  HOST = 'pi-head'
  PORT = 50008
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect((HOST, PORT))
    #s.send(data)
    send_msg(s, data)
    #print 'data sent to controller..'
  except socket.error, v:
    errorcode=v[0]
  s.close()
  #except socket.error, v:
  #  errorcode=v[0]
    #if errorcode==errno.ECONNREFUSED:
    #    print "Connection Refused"


def get_sync_data_socket():#MBList, policies, newAlloc
  global get_sync_data
  get_sync_data = False
  global policies
  global MBList
  global newAlloc
  newAlloc = {}

  HOST = '127.0.0.1'
  PORT = 50009
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)

  while True:
    #get_data =  False
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_sync_data - Connected by', addr
    #print 'before receiving sync data:'
    #print 'policies >>>>', len(policies)
    #print 'MBList >>>>', len(MBList)
    #print 'newAlloc >>>>', len(newAlloc)
    #data = ''
    #while 1:
      #buff = conn.recv(1024)
      #if not buff: break
      #data = data + buff
    #data = conn.recv(500000)
    data = recv_msg(conn)
    data_arr = pickle.loads(data)
    #print 'Received data:', repr(data_arr)
    policies = copy.deepcopy(data_arr[0])
    print 'policies >>>>', len(policies)
    MBList = copy.deepcopy(data_arr[1])
    print 'MBList >>>>', len(MBList)
    newAlloc = copy.deepcopy(data_arr[2])
    print 'newAlloc >>>>', len(newAlloc)
    get_sync_data = True

def migrateVMs():
  global get_sync_data
  global mig_end
  mig_end = False
  #get_sync_data = False
  while True:
    if get_sync_data == True:
      get_sync_data = False
      for vm,s in newAlloc.items():
	print vm,'->',s
        #migrateVM(net, vm, s)
      print 'Send data to controller after VM migration'
      #print 'after', flows
      #send_data_socket()
      #send_rSync_socket(False)
      mig_end = True
      #time.sleep(3)
      send_data_socket()
      send_rSync_socket(False)
    else:
      time.sleep(0.1)

def send_exp_done_socket():
  end_exp = True
  data = pickle.dumps(end_exp)

  HOST = 'pi-head'
  PORT = 50013
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  #print end_exp
  try:
    s.connect((HOST, PORT))
    s.send(data)
    #print 'send_exp_done_socket'
  except socket.error, v:
    errorcode=v[0]

  s.close()
  #except socket.error, v:
  #  errorcode=v[0]
    #if errorcode==errno.ECONNREFUSED:
    #    print "Connection Refused"



def get_msg_socket():
  global msg 
  #msg = False
  HOST = '127.0.0.1'
  PORT = 50010
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)
  
  while True:
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_msg_socket - Connected by', addr

    data = conn.recv(1024)
    data_arr = pickle.loads(data)
    msg = data_arr
    #print 'Received data:', repr(data_arr)
    #msg = copy.deepcopy(data_arr[0])

def get_ack_data():    
  global ack_data
  #msg = False
  HOST = '127.0.0.1'
  PORT = 50012
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)

  while True:
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_ack_data - Connected by', addr

    data = conn.recv(1024)
    data_arr = pickle.loads(data)
    ack_data = data_arr
    #print ack_data
    #print 'Received data:', repr(data_arr)
    #msg = copy.deepcopy(data_arr[0])


def send_data_10s():
  global msg
  msg = False
  while True:
    if msg == True:
      msg = False
      send_data_socket()
      send_topo_socket()
      send_rSync_socket(False)
    else:
      time.sleep(0.1)

def createTopo():
    #g = Graph()
    #graphs = None
    topo = MyTopo()
    topo.createSwitches()
    topo.createHosts()
    topo.createLinks()
    topo.createMBs()

    '''
    print 'check ids.....'
    print HostList
    print EdgeSwitchList
    print AggSwitchList
    print CoreSwitchList

    print 'h312',GetPodId('h312') 
    print 's1',GetPodId('s1')
    print 's9',GetPodId('s9')
    print 'h35',GetPodId('h35')
    print 's4',GetPodId('s4')
    print 's5',GetPodId('s5')
    print 's8',GetPodId('s8')
    print 's11',GetPodId('s11')
    '''

    global net
    CONTROLLER_IP = "150.204.50.231"
    CONTROLLER_PORT = 6633
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink, autoStaticArp=True, controller=None)
    net.addController( 'controller',controller=RemoteController,ip=CONTROLLER_IP,port=CONTROLLER_PORT)

    net.start()
    net.staticArp()



    global mymac_mb
    mymac_mb = {}
    for s in AllSwitches:
      for m,v in MBList.items(): 
	#mb = net.get(str(m))
	#l = topo.port(str(s),str(m))
	#if len(l) > 0:
	  #mymac_mb[str(mb.MAC())] = (int(str(s)[1:]),l[0])
	mymac_mb[str(m)] = (10, 0)
	  #print s,m,l,mb.MAC()
	  #print str(mb.MAC()), mymac_mb[str(mb.MAC())]
    '''
    for s in AllSwitches:
      for m,v in masses.items():
	if len(topo.port(str(s),str(m))) > 0:
          print s,m,topo.port(str(s),str(m)) 
    '''

    #createVMs(net)
    createPolicies()
    #createPolicies_()
    #createFlows()
    #dump_data(0)

    print 'VMs:',len(vms)    
    print 'MBs:', len(MBList)
    print 'Policies:', len(policies)/2, '- all:',len(policies)
    print 'Flows:', len(flows)/2, '- all:', len(flows)

    #print nodes
    #print edges
    #print graphs[0][0]
    #print graphs[0][1]
    #print graphs[0][2]
    #print graphs[0][3]


    #for k,v in AllHosts.items():
    #  for km,vm in MBList.items():
    #    print k,km,shortest_path(k,km, 0)


    #mb_type = ['FW', 'IPS', 'Proxy']

    
    global graphs
    mbtt = ['FW', 'IPS', 'Proxy']
    for p in permutations(mbtt):
      print (p[0],p[1],p[2])
      create_graph_flow((p[0],p[1],p[2]),p[0],p[1],p[2])
    

      #print graphs[1][0]
      #print shortest_path(str(0),str(1),(p[0],p[1],p[2]))
      #print shortest_distance(str(0),str(1),(p[0],p[1],p[2]))
    #print len(graphs) 
    #print graphs[0]

    #print graphs[('FW', 'IPS', 'Proxy')][0]
    #print graphs[('FW', 'IPS', 'Proxy')][1]
    #print graphs[('FW', 'IPS', 'Proxy')][2]
    #print graphs[('FW', 'IPS', 'Proxy')][3]


    send_data_socket()#send data to controller, don't run sync
    send_topo_socket()
    send_rSync_socket(False)

    CLI(net)
    '''
    print 'Starting flows before applying Sync policies:'
    for k,v in flows.items():
	#if k == 'f1':
	if str(k).startswith('b_') == False:
          startFlow(net, k)
    '''
    
    print 'send data to the controller and run sync'

    send_data_socket()#send data to controller and run sync
    send_rSync_socket(True)

    print 'data sent, getting data from controller for VM migration...'

    time.sleep(3)#to make sure all switches are registered

  
    while True:
      if mig_end == True:# and ack_data==True:
        print 'Starting flows after applying Sync policies:'
        #start all flows for 10s
        for k,v in flows.items():
	  if str(k).startswith('b_') == False:
	    print 'startFlow:', k
            #startFlow(net, k)
	break
      else:
        time.sleep(0.1)
    
    send_exp_done_socket()
    
    net.stop()
 
if __name__ == '__main__':
    setLogLevel('info')
    #print "hello"
    createTopo()
#topos = {'myfattreetopo': (lambda: MyTopo())}
