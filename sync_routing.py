
# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ryu.base import app_manager
from ryu.controller import mac_to_port
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import ether_types
from ryu.lib import mac
from ryu.lib import hub
 
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase
from ryu.topology import event, switches
from collections import defaultdict
from itertools import groupby
import numpy as np

import socket, threading, copy, time, struct, subprocess
import cPickle as pickle
import random
import csv
import os.path


from flow_stats import *

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
	#print len(data)
    return data
########################################


############### topology ###############

graphs = {}

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

    #route = shortest_path(src, dst, graph_name)
    #d = 0
    #for i in range(len(route)-1):
    #    d = d + get_distance(route[i], route[i+1], graph_name)
    #return d
    #return random.randint(2,18)  
    cost = CostBetween2Nodes(src,dst)
    return cost

########################################

############### for sync ###############


paths_db = {}
paths_db_mb = {}

installed_flows = []

CoreSwitchList = []
AggSwitchList = []
EdgeSwitchList = []

HostList = []

AllHosts = {}
MBList = {}
policies = {}
vms = {}
flows = {}
masses = {}

Groups = [] #o
ro = [] #o
newAllocAll = {} #output to topologyll



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

  if str(node).startswith('mb'):
    node = MBList[node][3]

  if IsCore(node):
    return -2


  if IsAgg(node):
    pod = (int(str(node)[1:]) - (num_core*num_group)) / num_core
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


  if str(node).startswith('mb'):
    node = MBList[node][3]

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
  if str(node1).startswith('mb'):
    node1 = MBList[node1][3]
  if str(node2).startswith('mb'):
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

def getActiveVms():
    vmids = []
    for k,v in flows.items():
        vmids.append(v[1])
        vmids.append(v[2])

    vmids = list(set(vmids))
    return vmids

def formVmFlowsTable(): # vm-flow(id, src, dst)
  VmFlowsDB = defaultdict(list)
  #vms[str(vmname)] = [pid, str(server), capacity, AllHosts[str(server)][0]]
  start_time=time.time() 
  print 'start formVmFlowsTable...'
  for vm,y in vms.items():
    for i,v in flows.items():
      if (v[1] == vm or v[2] == vm) and str(i).startswith('b_') == False:
	VmFlowsDB[vm].append([i,v[0],v[1],v[2],v[6]])#index, rate, src, dst, policy
  print 'end formVmFlowsTable in',time.time()-start_time

  return VmFlowsDB

def getVmFlows(vm, VmFlowsDB):
  k = []
  #for i,v in flows.items():
  #    if (v[1] == vm or v[2] == vm) and str(i).startswith('b_') == False:
  #        k.append([i,v[1],v[2]])
  #k = VmFlows[vm]
  #flows_vm=VmFlowsDB[vm]
  k = VmFlowsDB[vm]
  #for v in VmFlowsDB[vm]:
  #  k.append([v[0],v[2],v[3]]) # index, src, dst

  k.sort()
  VmFlows = list(k for k,_ in itertools.groupby(k))
  return VmFlows

def getNCVG(runnum, VmFlowsDB):
  del Groups[:]

  f_name_stat_CV = 'time_sync_cv.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
  f_cv=open(f_name_stat_CV, 'w')

  vmids = getActiveVms()
  i = 1
  while len(vmids) > 0:
      start_time = time.time()
      Group = []
      Group.append(vmids[0])
      VmFlows = getVmFlows(vmids[0], VmFlowsDB)
      #print 'length of flows size of',vmids[0],'is',len(VmFlows)
      while (len(VmFlows) > 0):
          fj = VmFlows[0]
          if fj[2] not in Group:
              vp = fj[2]
          elif fj[3] not in Group:
              vp = fj[3]
          else:
              VmFlows.remove(fj)
              continue
          Group.append(vp)
          k = VmFlows + getVmFlows(vp, VmFlowsDB)
          k.sort()
          VmFlows = list(k for k,_ in itertools.groupby(k))
          VmFlows.remove(fj)

      Groups.append(Group)
      vmids = [vm for vm in vmids if vm not in Group]
      end_time = time.time() - start_time
      f_msg = 'getNCVG,%s,%s,%s' % (i,len(Group),end_time)
      print f_msg
      f_cv.write(f_msg+"\n")
      i=i+1
  f_cv.close()
  #if len(Groups) > 0:
  #    print 'Printing Vm Groups...'
  #    for g in Groups:
  #        print g

def vmMigCost(vm, h):
  h = h.split('_', 1)[0]
  if vms[vm][1] != h:
    return 3
  else:
    return 0

def initialiseRo():
  global ro
  ro = {}
  for k,v in AllHosts.items():
    for k1,v1 in vms.items():
      ro[(k,k1)]=0

  #global ro
  #for k,v in AllHosts.items():
  #  for k1,v1 in vms.items():
  #    ro.append([k,k1,0])



def updateRo(s,v):
  ro[(s,v)]=ro[(s,v)]+1
  #for i in ro:
  #  if i[0] == s and i[1] == v:
  #    i[2] = int(i[2]) + 1
  #    break


def SPF(f, sh_paths_spf):
  #global graphs
  p = flows[f][6]
  #fr = flows[f][0]
  #vm_s = flows[f][1]
  #vm_d = flows[f][2]
  mblist = policies[p]
  mb1t = MBList[str(mblist[0])][2]
  mb2t = MBList[str(mblist[1])][2]
  mb3t = MBList[str(mblist[2])][2]
  #shortest_distance(g,'h1','mb1')
  #print (mb1t,mb2t,mb3t)

 
  #print 'before sh path'
  k = (mb1t,mb2t,mb3t)
  #for k,v in graphs.items():
  #  if k != 0 and k==(mb1t,mb2t,mb3t):
  #    print k
  #    sl = shortest_path(str(0),str(1),k)
      #sh_paths_spf[k]=sl
  #    break

  sl = sh_paths_spf[k]
  #sl = shortest_path(str(0),str(1),k)
  hs = sl[1]
  hd = sl[5]
  mlist = [sl[2],sl[3],sl[4]]
      #break      
      #print 'after sh path'
  #hs = 'h1'
  #hd = 'h2'
  #mlist = ['mb1', 'mb2', 'mb3']
  hs = hs.split('_', 1)[0]
  hd = hd.split('_', 1)[0]
  #print hs,hd,mlist
  return hs,hd,mlist


def getVmFlowsGroup(vm_group, VmFlowsDB):
  vms_flows = []

  for vm in vm_group:
    vms_flows = vms_flows + getVmFlows(vm, VmFlowsDB)

  vms_flows.sort()
  vms_flows = list(vms_flows for vms_flows,_ in itertools.groupby(vms_flows))

  return vms_flows

def phase_I(vm_group,runnum, sh_paths_spf, VmFlowsDB):
  #initialiseRo()
  #getNCVG()
  #if len(Groups) > 0:
  #global graphs
  #sh_paths_spf = {}

  #construct sh_paths_spf
  #for k,v in graphs.items():
  #  if k != 0:
  #    sl = shortest_path(str(0),str(1),k)
  #    sh_paths_spf[k]=sl 

 
  #start_time = time.time()
  vms_flows = getVmFlowsGroup(vm_group, VmFlowsDB)
  nb_flows = len(vms_flows)
  #print 'vms_flows:',vms_flows
  #print time.time()-start_time,'size of vms_flows is', len(vms_flows)
  for f in vms_flows:
    fk = f[0]
    pk = f[4]
    #print fk
    #pk = flows[fk][6]
    #print pk
    src,dst,mlist = SPF(fk, sh_paths_spf)
    #print src,dst,mlist
    #print 'policy ', pk,' before migration:',policies[pk]
    p_before = policies[pk]
    migratePolicy(fk,mlist,runnum)

    #if p_before != policies[pk]:
    #  print 'No migration of policy:',pk,policies[pk]
    #else:
    #  print 'Policy:',pk,'migrated from:',p_before,'to:',policies[pk]

    #print 'policy ', pk,' after migration:',policies[pk]
    vm_s = f[2] # flows[fk][1]
    vm_d = f[3] # flows[fk][2]
    updateRo(src,vm_s)
    updateRo(dst,vm_d)

  return nb_flows

def migratePolicy(f,mlist,runnum):
  p = flows[f][6]
  if str(p).startswith('b_') == True:
    print 'bad' 
  mblist0 = policies[p]
  f_c = flows[f][0]

  f_name_policies = 'policies.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'

  if mlist[0]!=mblist0[0] or mlist[1]!=mblist0[1] or mlist[2]!=mblist0[2]:
    if f_c <= MBList[mlist[0]][4]:
      mb1 = mlist[0]
      if mb1 != mblist0[0]:
        MBList[mblist0[0]][4] = float(MBList[mblist0[0]][4]) - float(f_c)
    else:
      mb1 = mblist0[0]

    if f_c <= MBList[mlist[0]][4]:
      mb2 = mlist[1]
      if mb2 != mblist0[1]:
        MBList[mblist0[1]][4] = float(MBList[mblist0[1]][4]) - float(f_c)
    else:
      mb2 = mblist0[1]

    if f_c <= MBList[mlist[0]][4]:
      mb3 = mlist[2]
      if mb3 != mblist0[2]:
        MBList[mblist0[2]][4] = float(MBList[mblist0[2]][4]) - float(f_c)
    else:
      mb3 = mblist0[2]

    if mlist[0]!=mblist0[0] or mlist[1]!=mblist0[1] or mlist[2]!=mblist0[2]:
      #print 'Policy',p,'is migrated'
      f_policies = open(f_name_policies, 'a')
      f_policies.write(str(p)+','+str(policies[p])+"\n")
      f_policies.close()
      policies[p] = [mb1, mb2, mb3]
      policies['b_'+str(p)] = [mb3, mb2, mb1]


def getVmFlows_cost(vm, VmFlowsDB):
  k = []
  #for i,v in flows.items():
  #    if (v[1] == vm or v[2] == vm) and str(i).startswith('b_') == False:
  #        k.append([v[0],v[1],v[2],v[6]])#flowrate, vm1, vm2, policy

  k = VmFlowsDB[vm]
  #for k,v in VmFlowsDB.items():
    
  if len(k)>0:
    k.sort()
    VmFlows = list(k for k,_ in itertools.groupby(k))
  VmFlows = k
  return VmFlows


def calcTotCommCost(vm, s, VmFlowsDB):
  #start_time = time.time()
  vmFlows = getVmFlows_cost(vm, VmFlowsDB)
  #outgoing traffic from vm in s
  o = 0
  i = 0
  
  if len(vmFlows)>0:
    for f in vmFlows:
      if f[2] == vm:
        o = float(o) + float(f[1]) * float(shortest_distance(s,policies[f[4]][0],0))
      elif f[3] == vm:
        i = float(i) + float(f[1]) * float(shortest_distance(s,policies[f[4]][2],0))

  #print 'calcTotCommCost of:',vm, s,' is:', int(o)+int(i)
  #print time.time()-start_time,'calcTotCommCost, loop of',len(vmFlows),'flows'
  return float(o)+float(i)

def utilityOfMig(vm,s,totcomcost_vm, VmFlowsDB):
  vm_s = vms[vm][1]
  #vm_s = ''
  #for k,v in vms.items():
  #  if k == vm:
  #    vm_s = v[1]

  return totcomcost_vm - calcTotCommCost(vm,s, VmFlowsDB) - vmMigCost(vm,s)

def obtainPrefList():
  global prefList
  prefList = {}
  #print len(ro)
  #for item,row in ro.items():
  #  print item[0],item[1], row

  #change ro structure
  ro_ = defaultdict(list)
  for k,v in ro.items():
    ro_[k[0]].append([k[1],v])
  #print ro_,'done'
  

  for s,v in AllHosts.items():
    l = []
    l = ro_[s]
    #for item,row in ro.items():
    #  if item[0] == s:
    #	item[0],item[1],row
    #    l.append([item[1],row])
    #for item,row in ro_.items():
    #  if item == s:
    #    l.append()

    l.sort(key=lambda x: x[1], reverse=True)
    #print l
    lv = []
    for i in l:
      lv.append(i[0])
    prefList[s] = lv
    #print 'server',s,'done'


def initialiseBList(vm_group):
  #global blackList
  blackList = {}
  for vm in vm_group:
    blackList[vm] = []

  return blackList

def checkConnectivity(vm_s, server):
  #vm_s = vms[vm][1]
  if((IsAgg(vm_s) or IsAgg(server)) or (IsCore(vm_s) or IsCore(server))):
    #print 'same agg or same core'
    return True

  pod1 = GetPodId(vm_s)
  pod2 = GetPodId(server)
  #print vm_s,pod1,server,pod2

  #nodes within the same pod can reach each other
  if(pod1 == pod2):
    #print 'same pod'
    return True

  numPod = 12 #k is fixed

  if((pod1<numPod/2 and pod2<numPod/2) or (pod1>=numPod/2 and pod2>=numPod/2)):
    #print pod1,pod2,numPod/2
    return True
  else:
    #print 'false---------!!'
    return False

  return True

def getAvailableServer(vm, blackList_vm):
  vm_c = vms[vm][2]
  vm_s = vms[vm][1]
  list_srvs = []
  for k,v in AllHosts.items():
    if vm_s==k:
      list_srvs.append(k)
    elif (checkConnectivity(vm_s,k)==True) and (v[2] >= vm_c):
      list_srvs.append(k)
  
  return list_srvs

def IsOkVmOnServer(vm,vmFlows,server):
  #vmFlows = getVmFlows_cost(vm)
  node=None
  if len(vmFlows)>0:
    for f in vmFlows:
      if f[2] == vm:
	node = policies[f[4]][0]
      else: 
	node = policies[f[4]][2]
      if(checkConnectivity(server,node)==False):
        return False
  return True
    

def getMaxUtility(vm, blackList_vm, totcomcost_vm, VmFlowsDB):
  #obtain S(v) \ blacklist
  #vm_s = vms[vm][1]
  #vm_c = vms[vm][2]
  #list_srvs = []
  #start_time=time.time()
  #for k,v in AllHosts.items():
  #  if (v[2] >= vm_c and k not in blackList_vm and ) or vm==vm_s:
  #    list_srvs.append(k)
  #print 'went through all servers takes:',time.time()-start_time
  #determine sj
  list_srvs = getAvailableServer(vm, blackList_vm)
  vmFlows = getVmFlows_cost(vm, VmFlowsDB)
  max_ = -1000000000
  sj = None
  #print 'list_srvs in getMaxUtility:',list_srvs
  #start_time=time.time()
  for s in list_srvs:
    #print 'utilityOfMig(g,vm,s):',vm,s,utilityOfMig(g,vm,s)
    if IsOkVmOnServer(vm,vmFlows,s)==False:
      #print 'IsOkVmOnServer'
      continue
    u = utilityOfMig(vm,s,totcomcost_vm,VmFlowsDB)
    #print 'utilityOfMig=',u
    if float(u) >= float(max_) and float(u)>=0:
      max_ = u
      sj = s
  #print 'list_srvs size',len(list_srvs),'getMaxUtility takes:',time.time()-start_time

  #print 'sj:',sj
  return sj

def checkSrvCapacity(sj, newhvm, hostedVMs):
  #vm_s = vms[vm][1] #is always different to s
  #vm_c = vms[vm][2]
  #time_start = time.time()
  #hostedVMs = obtainHostedVms
  #print sj,hostedVMs
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
      vm_c = vm_c + float(vms[i][2])

  s_s = AllHosts[sj][2]
  if s_s >= vm_c:
    #print time.time()-time_start,'end checkSrvCapacity',sj
    return True
  else:
    #print time.time()-time_start,'end checkSrvCapacity',sj
    return False

def getUnprocessedVms(newAlloc):
  l = []
  for k,v in newAlloc.items():
    if v == 0:
      l.append(k)
  return l

def lastVmInPrefList(sj, newHostedVms):
  lj = prefList[sj]
  lj1 = [x for x in lj if x in newHostedVms]
  if len(lj1) > 0:
    return lj1[-1]
  else:
    return -1

def obtainHostedVms():
  #global hostedVMs
  hostedVMs = {}
  for h,hv in AllHosts.items():
    l = []
    for k,v in vms.items():
      if v[1] == h:
        l.append(h)
    hostedVMs[h] = l

  return hostedVMs

def phase_II(vm_group, hostedVMs, VmFlowsDB):
  #global newAlloc
  #global exitAlloc
  #global hostedVMs
  #global blackList
  global newAllocAll

  #start_time = time.time()
  #newAlloc = {}i

  #print 'vm_group',vm_group


  #print vms['vm1']

  newAlloc = {}
  for v in vm_group:
    newAlloc[v] = 0

  newAlloc[1] = 1 #bug!!
  #print 'newAlloc',newAlloc
    #print v
  #print start_time-time.time(), 'end initialise newAlloc dict'
  #run obtainHostedVms - used in checkSrvCapacity(..) - begining of Sync

  newHostedVms = defaultdict(list)

  #obtainPrefList()#outside phase_II - after phase_I(..)

  #start_time = time.time()
  blackList = initialiseBList(vm_group)
  #print time.time()-start_time, 'end initialise Black List'
  #l = getUnprocessedVms(newAlloc)
  while len(getUnprocessedVms(newAlloc)) > 0:
    #start_time = time.time()
    l = getUnprocessedVms(newAlloc)
    #print l
    #print time.time()-start_time, 'end getUnprocessedVms',l
    vm = l[0]
    #print 'vm',vm
    #print 'before getMaxUtility:',vm, blackList[vm]

    #calculate calcTotCommCost(vm,vm_s) to avoid recalculating it each time# vm and vm_s are fixed
    
    vm_s = vms[vm][1]
    #print vm, vm_s, vms[vm]
    totcomcost_vm = calcTotCommCost(vm,vm_s,VmFlowsDB)
    #start_time = time.time()
    sj = getMaxUtility(vm,blackList[vm],totcomcost_vm,VmFlowsDB)
    if sj==None:
      newAlloc[vm] = vm_s
      continue
    #print time.time()-start_time, 'end getMaxUtility',vm,blackList[vm],sj

    newAlloc[vm] = sj
    #vm_on_s_g: group of all vm on server sj before and after migrationxUtility
    newHostedVms[sj].append(vm)
    #print 'len of newHostedVms[sj] before checkSrvCapacity:', sj, len(newHostedVms[sj])
    if checkSrvCapacity(sj, newHostedVms[sj], hostedVMs) == False:
      #start_time = time.time()
      while True:
        vk = lastVmInPrefList(sj, newHostedVms[sj])
        newAlloc[vk] = None
        newHostedVms[sj].remove(vk)
        best_rejected = vk
        if checkSrvCapacity(sj, newHostedVms[sj], hostedVMs) == True:
          break
      #print time.time()-start_time, 'end checkSrvCapacity is false',sj,newHostedVms[sj],'while true ended'
      #print 'best_rejected:',best_rejected
      #print 'blackList:',blackList
      #print 'prefList[sj]:',prefList[sj]
      #start_time = time.time()
      for vk_ in [x for x in prefList[sj] if x in vm_group]:
        #print 'vk_:',vk_
        if vk_ != best_rejected:
          blackList[vk_].append(sj)
        elif vk_ == best_rejected:
          blackList[vk_].append(sj)
          break
      #print time.time()-start_time, 'end of for vk in preflist and vm_group'
      #print 'size blackList:', len(blackList)
    #print time.time()-start_time, vm, 'is processed'
  #existAlloc = {}
  #for k,v in vms.items():
  #  if k in vm_group:
  #    existAlloc[k] = v[1]
  #print 'Exisiting Allocation:', existAlloc
  #newAllocNow = {}
  #for k,v in newAlloc.items():
  #  if k in vm_group:
  #    newAllocNow[k] = v
  #print 'New Allocation for group:', newAllocNow
  #for vm,s in newAlloc.items():
  #  migrateVM(net, vm, s)

    for vm,s in newAlloc.items():
      newAllocAll[vm]=s

def send_sync_data_socket():
  newAllocAll = {}
  l = [policies, MBList, newAllocAll]
  #l = [AllHosts]
  data = pickle.dumps(l)

  HOST = 'ubuntu-server-02'
  PORT = 50009
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect((HOST, PORT))
    #s.send(data)
    send_msg(s, data)
    #print 'data sent to controller..'
  except socket.error, v:
    errorcode=v[0]
    print errorcode
  s.close()
  #except socket.error, v:
  #  errorcode=v[0]
    #if errorcode==errno.ECONNREFUSED:
    #    print "Connection Refused"


def get_rSync_socket():
  global rSync
  #msg = False
  HOST = '127.0.0.1'
  PORT = 50011
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)

  while True:
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_rSync_socket - Connected by', addr

    data = conn.recv(1024)
    data_arr = pickle.loads(data)
    rSync = data_arr
    #print 'rSync=',rSync
    
    #print 'Received data:', repr(data_arr)
    #msg = copy.deepcopy(data_arr[0])


def send_msg_socket():
  msg = True
  data = pickle.dumps(msg)

  HOST = 'ubuntu-server-02'
  PORT = 50010
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

def send_ack_data():
  ack_data = True
  data = pickle.dumps(ack_data)

  HOST = 'ubuntu-server-02'
  PORT = 50012
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

def get_exp_done_socket():
  global end_exp
  global f_ryu_vmstat_name
  global f_ryu
  #msg = False
  HOST = '127.0.0.1'
  PORT = 50013
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)

  while True:
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_exp_done_socket - Connected by', addr

    data = conn.recv(1024)
    data_arr = pickle.loads(data)
    end_exp = data_arr
    #print end_exp
    if end_exp:
	print 'end_exp is', end_exp
	#f_ryu.close()
	kill = subprocess.Popen(['sh', 'kill.sh', 'vmstat'])
        kill.wait() 
	#kill = subprocess.Popen(['sh', 'kill.sh', 'netspeed'])
	#kill.wait()
	
    #print 'Received data:', repr(data_arr)
    #msg = copy.deepcopy(data_arr[0])



def get_topo_socket():
  global get_topo
  get_topo = False
  #global nodes #input
  #global edges #input
  #global distances #input
  #global _distance #input
  global graphs
  global mymac_mb

  HOST = '127.0.0.1'
  PORT = 50008
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)

  while True:
    #get_topo = False
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_topo_socket - Connected by', addr

    #data = conn.recv(50000)
    data = recv_msg(conn)
    data_arr = pickle.loads(data)
    #print 'Received data:', repr(data_arr)
    #nodes = copy.deepcopy(data_arr[0])
    #print '>>>>', len(nodes)
    #edges = copy.deepcopy(data_arr[1])
    #print '>>>>', len(edges)
    #distances = copy.deepcopy(data_arr[2])
    #print '>>>>', len(distances)
    #_distance = copy.deepcopy(data_arr[3])
    #print '>>>>', len(_distance)
    graphs = copy.deepcopy(data_arr[0])
    mymac_mb = copy.deepcopy(data_arr[1])
    #print 'mymac_mb >>>>', len(mymac_mb)
    print 'size graphs:', len(graphs)
    global mymac
    for src,v in mymac_mb.items():
      #print src,v
      if src not in mymac.keys():
        mymac[src]=( int(v[0]),  int(v[1]))

      #mymac[str(k)] = (int(v[0]),int(v[1]))
    #print 'mymac:',mymac
    '''
    global paths_db_mb
    policy_found=False
    for k,v in policies.items():
      mb1 = MBList[v[0]][1]
      mb2 = MBList[v[1]][1]
      mb3 = MBList[v[2]][1]
      print k,mb1,mb2,mb3
      print policies
      p2=None
      p3=None
      p2 = get_path(mymac[mb1][0] , mymac[mb2][0], mymac[mb1][1], mymac[mb2][1],policy_found)
      p3 = get_path(mymac[mb2][0], mymac[mb3][0], mymac[mb2][1], mymac[mb3][1],policy_found)
      paths_db_mb[k]=[p2,p3]
      print 'path added to policy', k      
    '''

    get_topo = True

def runSync():
  global get_topo
  global get_data
  global rSync
  global sync_data_ready
  global f_ryu_vmstat_name
  rSync = False
  sync_data_ready = False
  #initialise
  #get_topo = False
  #get_data = False
  #rSync = False
  while True:
    if get_data==True and get_topo==True and rSync==True:
      #print get_data,get_topo,rSync
      get_data = False
      get_topo = False
      rSync = False
      #print 'nodes\t:',             len(nodes)
      #print 'edges\t:',             len(edges)
      #print 'distances\t:',     len(distances)
      #print '_distance\t:',     len(_distance)
      #print 'AllHosts\t:',     len(AllHosts)
      #print 'MBList\t:',     len(MBList)
      #print 'policies\t:',     len(policies)
      #print 'vms\t:',             len(vms)
      #print 'flows\t:',      len(flows)

      if not os.path.exists("/tmp/sync/"):
            os.makedirs("/tmp/sync/")

      if start_exp == True:
	#stop vmstat of ryu
	print 'start_exp=True'
      	kill = subprocess.Popen(['sh', 'kill.sh', 'vmstat'])
      	kill.wait()
	

      runs = 10 #run sync 3 times 
      print len(flows)
      VmFlowsDB = createFlows()
      print len(flows)
      print len(VmFlowsDB)
      for runnum in range(1,runs+1):



	#run obtainHostedVms - used in checkSrvCapacity(..) - begining of Sync
	hostedVMs = obtainHostedVms()
	#VmFlowsDB = formVmFlowsTable() 
	#print len(VmFlowsDB)
	#print VmFlowsDB['vm1']

        #obtainPrefList()#outside phase_II - after phase_I(..)

        f_name = 'vmstat_sync.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
        f_name_stat_phasei = 'time_sync_pi.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
        f_name_stat_phaseii = 'time_sync_pii.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
        #f_name_policies = 'policies.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
        f_name_vms = 'vms.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
        f_name_flows = 'flows.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'_'+str(len(vms))+'.'+str(runnum)+'.dat'
        #f_name_stat_CV = 'time_sync_cv.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'.dat'

        #f_cv=open(f_name_stat_CV, 'w')
        f_pi = open(f_name_stat_phasei, 'w')
        f_pii = open(f_name_stat_phaseii, 'w')

        f_vmstat = open(f_name, 'w')
        f_vmstat.close()


	#f_policies = open(f_name_policies, 'w')
        #f_policies.close()

	f_vms = open(f_name_vms, 'w')
	f_flows = open(f_name_flows, 'w')

	old_policies = copy.deepcopy(policies)

        print 'Apply Sync policy, run number:',runnum
        print 'Initialise Ro'
        vmstat = subprocess.Popen(['sh', 'vmstat.sh'] + [f_name])
        start_time_sync = time.time()
        initialiseRo()
        f_vmstat = open(f_name, 'a')
        f_vmstat.write(" z\n")
        f_vmstat.close()

        #print 'Get communicating VMs'
        #subprocess.Popen(['/bin/bash', '-c', '>'] + [f_name])
        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- start ---------" >> vmstat.txt'])

        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- Get communicating VMs: START ---------" >> vmstat.txt'])
        #vmstat = subprocess.Popen(['sh', 'vmstat.sh', 'vmstat.txt'])
        #f_name = 'vmstat_'+str(len(flows)/2)+'_'+str(len(policies)/2)+'.txt'
        #vmstat = subprocess.Popen(['sh', 'vmstat.sh'] + [f_name])
        #start_time = time.time()
        getNCVG(runnum, VmFlowsDB)
        #f_msg = 'getNCVG,%s,%s,%s' % (len(Groups),p,time.time() - start_time)
        #p = p + 1
        #print f_msg
        #f_cv.write(f_msg+"\n")
        #f_cv.close()
        f_vmstat = open(f_name, 'a')
        f_vmstat.write(" z\n")
        f_vmstat.close()

        kill = subprocess.Popen(['sh', 'kill.sh', 'vmstat'])
        kill.wait()
        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- Get communicating VMs: DONE ---------" >> vmstat.txt'])
        #print("--- getNCVG %s seconds ---" % (time.time() - start_time_sync))

        i = 1
        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- Phase I: START ---------" >> vmstat.txt'])
        #vmstat = subprocess.Popen(['sh', 'vmstat.sh', 'vmstat.txt'])
        vmstat = subprocess.Popen(['sh', 'vmstat.sh'] + [f_name])
        start_time_phase_I = time.time()
	
	#shortest paths of flow graphs
	global graphs
  	sh_paths_spf = {}
	#construct sh_paths_spf
	for k,v in graphs.items():
	  if k != 0:
	    sl = shortest_path(str(0),str(1),k)
	    sh_paths_spf[k]=sl

        for gr in Groups:
	  #print 'phase_I started of group',i,'/',len(Groups)
	  start_time = time.time()
          nb_flows = phase_I(gr,runnum,sh_paths_spf, VmFlowsDB)
	  end_time = time.time() - start_time
	  f_msg = 'phase_I,%s,%s,%s,%s' % (i,len(gr),end_time,str(nb_flows))
	  print f_msg
	  f_pi.write(f_msg+"\n")
          #print("--- phase_I %s seconds ---" % (time.time() - start_time))
	  i = i + 1
        end_time = time.time() - start_time_phase_I
        f_msg = 'phase_I_all,%s,%s,%s' % (len(Groups),len(Groups),end_time)
        print f_msg
        f_pi.write(f_msg+"\n")
        f_pi.close()
        f_vmstat = open(f_name, 'a')
        f_vmstat.write(" z\n")
        f_vmstat.close()



        kill = subprocess.Popen(['sh', 'kill.sh', 'vmstat'])
        kill.wait()

        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- Phase I: DONE ---------" >> vmstat.txt'])


        i = 1
        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- Phase II: START ---------" >> vmstat.txt'])
        #vmstat = subprocess.Popen(['sh', 'vmstat.sh', 'vmstat.txt'])
	obtainPrefList()
        vmstat = subprocess.Popen(['sh', 'vmstat.sh'] + [f_name])
        start_time_phase_II = time.time()
        for gr in Groups:
	  #print 'phase_II started of group',i,'/',len(Groups)
	  start_time = time.time()
          phase_II(gr, hostedVMs, VmFlowsDB)
	  end_time = time.time() - start_time
          #print("--- phase_II %s seconds ---" % (time.time() - start_time))
	  f_msg = 'phase_II,%s,%s,%s' % (i,len(gr),end_time)
	  print f_msg
	  f_pii.write(f_msg+"\n")	
	  i = i + 1
        end_time = time.time() - start_time_phase_II
        f_msg = 'phase_II_all,%s,%s,%s' % (len(Groups),len(Groups),end_time)
        print f_msg
        f_pii.write(f_msg+"\n")

        f_msg = 'Sync,%s,%s,%s' % (len(Groups),len(Groups),time.time() - start_time_sync)
        print f_msg
        f_pii.write(f_msg+"\n")

        f_vmstat = open(f_name, 'a')
        f_vmstat.write(" z\n")
        f_vmstat.close()
      
        f_pii.close()

        kill = subprocess.Popen(['sh', 'kill.sh', 'vmstat'])
        kill.wait()
      
        #subprocess.Popen(['/bin/bash', '-c', 'echo "--------- Phase II: DONE ---------" >> vmstat_sync.7_10.csv'])

        #print("--- Sync %s seconds ---" % (time.time() - start_time_sync))
	#for k,v in policies.items():
	#  if v!=old_policies[k]:
	#    f_policies.write(str(k)+","+str(v)+",migrated\n")

	#newAllocAll = {}
	for vm,s in newAllocAll.items():
	  f_vms.write(str(vm)+","+str(s)+"\n")

	for k,v in flows.items():
	  if v[1] in newAllocAll.keys() or v[2] in newAllocAll.keys():
	    f_flows.write(str(k)+"\n")


	kill = subprocess.Popen(['sh', 'kill.sh', 'vmstat'])
        kill.wait()

	f_vms.close()
	f_flows.close()

        print 'Sync done.'
	time.sleep(10)
        
        #print 'Send Sync data and perform VM migration'
        sync_data_ready = True
        #send_sync_data_socket()
        #print 'Sync data sent to perform VM migration'
    elif sync_data_ready:
      sync_data_ready = False
      send_sync_data_socket()
      print 'Sync data sent to perform VM migration'
      if start_exp == True:
        vmstat = subprocess.Popen(['sh', 'vmstat.sh'] + [f_ryu_vmstat_name])
    else:
      time.sleep(0.1)

########################################


#switches
switches = []

#mymac[srcmac]->(switch, port)
mymac={}
mymac_mb={}

#adjacency map [sw1][sw2]->port from sw1 to sw2
adjacency=defaultdict(lambda:defaultdict(lambda:None))

#weight dictionary: weight[(s1,s2)]->w
weight={}


def populate_sync_data():
  #read csv files and populate mbs, policies, and hosts
  AllHosts.clear()
  MBList.clear()
  policies.clear()
  vms.clear()
  flows.clear()
  with open("/tmp/sync/hosts.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      #print row
      AllHosts[row[0]] = [row[1], row[2]]
  with open("/tmp/sync/mbs.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      #print row
      MBList[row[0]] = [row[1], row[2], row[3], row[4]]
  with open("/tmp/sync/policies.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      #print row
      policies[row[0]] = [row[1], row[2], row[3]]
      policies["b_"+row[0]] = [row[3], row[2], row[1]]
  with open("/tmp/sync/vms.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      #print row
      vms[row[0]] = [row[1], row[2], row[3]]
  with open("/tmp/sync/flows.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      #print row
      flows[row[0]] = [row[1], row[2], row[3],  row[4], row[5], row[6], row[7], 'd']
      flows["b_"+row[0]] = [row[1], row[3], row[2],  row[5], row[4], row[6], "b_"+row[7], 'i']


########### to create flows and vms at the controller ############

def createVM(server, vmname, capacity):#no net
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


def percentage(percent, whole):
  return (percent * whole) / 100.0


def createVMs():#no net
    hosts = []

    global vm_tenants
    vm_tenants = defaultdict(list)


    global group_sizes
    group_sizes = []    

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
    current_size = None
    tid = 1
    while i < nvm:#len(AllHosts):
        i = i + 1
        if i > 2*nvm:
            break
        server = random.choice(hosts)
        c = random.choice(vm_c)
        vmname = "vm" + str(j)
        #createVM(net, server, vmname, c)
        if createVM(server, vmname, c) == 0:#no net
            if j==1:#initialise current_size
	      #current_size = random.randint(int(nvpt-percentage(25,nvpt)),int(nvpt+percentage(25,nvpt)+1))
	      current_size = abs(int(np.random.normal(loc=nvpt, scale=495, size=None)))%1000
	      #current_size = int(np.random.normal(loc=nvpt, scale=10, size=None))
	      #current_size=5
	      group_sizes.append(current_size)
	      #print 'current_size',current_size
	      #time.sleep(5)
	    if len(vm_tenants[tid]) < current_size:#add vm to tid
	      vm_tenants[tid].append(vmname)
	    else:#move to another tid
	      #current_size = random.randint(int(nvpt-percentage(25,nvpt)),int(nvpt+percentage(25,nvpt)+1))
	      #current_size = int(np.random.normal(loc=nvpt, scale=10, size=None))
	      current_size = abs(int(np.random.normal(loc=nvpt, scale=495, size=None)))%1000
	      #current_size=5
	      group_sizes.append(current_size)
	      #print 'current_size',current_size
              #time.sleep(5)
	      tid = tid + 1
	      #print len(vm_tenants[tid-1]),tid-1, current_size


	    j = j + 1
    #for k,v in vm_tenants.items():
    #  print k, v
    #f_v.close()
    for k,v in vms.items():
      if k==1:
        print '------------------------------------------------------------'


def createFlows():
    VmFlowsDB = defaultdict(list)

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

    while fid < nf+1:
        #if i != a:
        #if a > len(vms)*4:
        #  break
        j = j + 1
        if j > nf*2:
	    print 'break'
            break
        #print len(vms)*4,a
        policy = random.choice(pcids)
        #proto = random.choice(protos)
        proto = 'tcp'
        #vmids_ = vmids[:]
        #vm1 = random.choice(vmids_)
        #vmids_.remove(vm1)
        #vm2 = random.choice(vmids_)
	while True:
	  tid = random.choice(vm_tenants.keys())
	  if len(vm_tenants[tid])>0:
	    break
        vm1 = random.choice(vm_tenants[tid])
        #vm1 = "vm"+str(v)
	vm2 = random.choice(vm_tenants[tid])
        #vm2 = "vm"+str(v)
	#print 'flow in',tid, vm1,vm2
        flowid = "f" + str(fid)
        #flowrate = random.randint(fmin, fmax)#Kbit/s
        flowrate=1
        issameip = 0
        if vms[vm1][3] == vms[vm2][3]:
            issameip = 1
        if createFlow(flowid, flowrate, vm1, vm2, s_port, d_port, proto, policy, issameip) == 0:
            print 'created flow:',flowid, ' between:',vm1,vm2, ' with policy:',policy
            VmFlowsDB[vm1].append([flowid,flowrate,vm1,vm2,policy])#index, rate, src, dst, policy
	    VmFlowsDB[vm2].append([flowid,flowrate,vm1,vm2,policy])#index, rate, src, dst, policy
	    fid = fid + 1
            if proto != 'icmp':
                s_port = s_port + 1
                d_port = d_port + 1
    #f_f.close()
    #print [vm_tenants.keys()]
    
    return VmFlowsDB

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



##################################################################


def get_data_socket():
  global get_data
  #global rSync
  get_data =  False
  
  #global ack_data
  #ack_data = False
  global flows#built here
  global policies
  global vms#built here
  global MBList
  global AllHosts
  global masses

  #needed to calculate cost between two nodes
  global CoreSwitchList
  global AggSwitchList
  global EdgeSwitchList
  global HostList

  HOST = '127.0.0.1'
  PORT = 50007
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((HOST, PORT))
  s.listen(5)
  data = ''
  while True:
    #print 'I am listening'
    conn, addr = s.accept()
    print 'get_data_socket - Connected by', addr
    #while 1:
    #  buff = conn.recv(1024)
    #  if not buff: break
    #  data = data + buff
    data = recv_msg(conn)
    #print recv_msg(s)
    #data = conn.recv(500000)
    #data_arr = ''
    data_arr = pickle.loads(data)
    #print data_arr
    #print 'Received data:', repr(data_arr)
    AllHosts = copy.deepcopy(data_arr[0])
    #print '>>>>', len(AllHosts)
    MBList = copy.deepcopy(data_arr[1])
    #print '>>>>', len(MBList)
    policies = copy.deepcopy(data_arr[2])
    #print 'number of policies >>>', len(policies)
    masses = copy.deepcopy(data_arr[3])

    CoreSwitchList = copy.deepcopy(data_arr[4])
    AggSwitchList = copy.deepcopy(data_arr[5])
    EdgeSwitchList = copy.deepcopy(data_arr[6])

    HostList = copy.deepcopy(data_arr[7])
    #vms = copy.deepcopy(data_arr[3])
    #print 'number of vms >>>', len(vms)
    #flows = copy.deepcopy(data_arr[4])
    #print 'number of flows >>>',len(flows)
    #print flows
    #rSync = copy.deepcopy(data_arr[5])
    #print 'rSync=',rSync, data_arr[5]
    #print rSync
    #print '>>>>', flows
    #print 'Received AllHosts:', repr(data_arr[0])
    #print 'Received MBList:', repr(data_arr[1])
    #print 'Received policies:', repr(data_arr[2])
    #print 'Received vms:', repr(data_arr[3])
    #print 'Received flows:', repr(data_arr[4])
    #ack_data = True

    send_ack_data()
    if len(vms)==0 or len(flows)==0:
      createVMs()
      #createFlows()
    get_data = True
    #print get_data


  conn.close()



def get_host(mac):
  for k,v in AllHosts.items():
    if v[1]==mac:
      return k

def get_host_ip(ip):
  for k,v in AllHosts.items():
    if v[0] == ip:
      return k

def get_mb_s_t(mac):
  for k,v in MBList.items():
    if v[1]==mac:
      return (v[2],v[3])

def get_ip(vm):
  h = vms[vm][1]
  ip = AllHosts[h][0]
  return ip

def proto_code(proto):
  if proto == 'icmp':
    return 1
  elif proto == 'tcp':
    return 6
  elif proto == 'udp':
    return 17

def get_policy(ip_src, ip_dst, port_src, port_dst, proto):
#def get_policy(mac1, mac2):
  #populate_sync_data()
  #get_data_socket()
  p = ''
  f = ''
  mb_list = []
  #print 'flows in get_policy>>>>>>>>>>>>>>>>>>>',flows
  for k,v in flows.items():
    if proto == proto_code(v[5]):
      if proto == 6 or proto == 17:
	#print '>>>>>>before condition>>>>>>>',k,str(v[8]),str(v[9]), v[3], v[4]
        if ip_src == str(v[8]) and ip_dst == str(v[9]) and str(port_src) == str(v[3]) and str(port_dst) == str(v[4]):
	  #print '>>>>>>after condition>>>>>>>',k,str(v[8]),str(v[9]), v[3], v[4]
	  #print 'ip_src == get_ip(v[1]).......', get_ip(v[1])
          #if v[7] == 'd':
	  #  print 'in v[7] == \'d\'',v[7], 'port_dst', v[4]
          #  if str(port_dst) == v[4]:#coz we dont know source port in iperf3...
	  #    p = v[6]
	      #print 'in port_dst == v[4]..........!!!',v[4]
          #else:
          #  if str(port_src) == v[3]:
	  #    p = v[6]
	  #    print 'in port_src == v[3]..........!!!',v[3]
	  p=v[6] 
	  f = k
      else:
        if ip_src == str(v[8]) and ip_dst == str(v[9]):
	  p = v[6]
	  f = k

  if p:
    mb1 = policies[p][0]
    mb2 = policies[p][1]
    mb3 = policies[p][2]
    mb_list = [MBList[mb1][1], MBList[mb2][1], MBList[mb3][1]]
  '''
  for k,v in policies.items():
    if get_host(mac1) == v[0] and get_host(mac2) == v[1]:
      mb_list = [mbs[v[2]][1], mbs[v[3]][1], mbs[v[4]][1]]
  '''
  return p,mb_list,f

def update_mymac_mbs():
  for k,v in MBList.items():
    if v[1] not in mymac.keys():
      mymac[v[1]] = (int(v[3][1:]),5)
    
def get_mac(dpid, in_port):
  #mymac[src]=( dpid,  in_port)
  for k, v in mymac.items():
      if v == (dpid,in_port):
          return k

'''
def clean_path(p):
  pnew = []
  for i in range(len(p)-1):
    if p[i][0]==p[i+1][0]:
      item = (p[i][0],p[i][1],p[i+1][2])
      pnew.append(item)
    elif p[i-1][0]!=p[i][0] and p[i][0]!=p[i+1][0]:
      pnew.append(p[i])
      if i+1==len(p)-1:
        pnew.append(p[i+1])
  return pnew
'''
def clean_path(p):
 pnew = []
 i = 0
 while i < len(p):
  if i < len(p)-1:
   if p[i][0]==p[i+1][0] and p[i][2]==p[i+1][1]:
    item = (p[i][0],p[i][1],p[i+1][2])
    pnew.append(item)
    i = i + 2
   else:
    item = p[i]
    pnew.append(item)
    i = i + 1
  else:
   item = p[i]
   pnew.append(item)
   i = i + 1  
 return pnew

def minimum_distance(distance, Q):
  min = float('Inf')
  node = 0
  #print 'Q=',Q
  for v in Q:
    #print v, distance[v], min
    if distance[v] < min:
      min = distance[v]
      node = v
  return node

def get_path (src,dst,first_port,final_port,policy_found):
  #Dijkstra's algorithm
  #if policy_found:
  #  print "get_path is called, src=",src," dst=",dst, " first_port=", first_port, " final_port=", final_port
  distance = {}
  previous = {}

  for dpid in switches:
    distance[dpid] = float('Inf')
    previous[dpid] = None

  distance[src]=0
  Q=set(switches)
  #print "Q=", Q

  while len(Q)>0:
    u = minimum_distance(distance, Q)
    #u = minimum_distance(distance, Q)
    #print '>>>>>>>>>>>>>>>>>>>>>>', u
    #if u in Q:
    Q.remove(u)

    for p in switches:
      if adjacency[u][p]!=None:
        w = weight[(u,p)]
        if distance[u] + w < distance[p]:
          distance[p] = distance[u] + w
          previous[p] = u

  r=[]
  p=dst
  r.append(p)
  q=previous[p]
  while q is not None:
    if q == src:
      r.append(q)
      break
    p=q
    r.append(p)
    q=previous[p]
 
  r.reverse()
  if src==dst:
    path=[src]
  else:
    path=r
  # Now add the ports
  r = []
  in_port = first_port
  for s1,s2 in zip(path[:-1],path[1:]):
    out_port = adjacency[s1][s2]
    r.append((s1,in_port,out_port))
    in_port = adjacency[s2][s1]
  r.append((dst,in_port,final_port))
  return r


def get_mac(dpid, in_port):
  #mymac[src]=( dpid,  in_port)
  for k, v in mymac.items():
      if v == (dpid,in_port):
          return k


class ProjectController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ProjectController, self).__init__(*args, **kwargs)
        print "hello ryu"
        global start_exp
        global end_exp
	#global f_ryu
	global f_ryu_vmstat_name
	#f_ryu = None
	f_ryu_vmstat_name = None
	#f_ryu = None
	#f_ryu=open('ryu_'+str(len(flows)/2)+'.csv', 'w')
	#f_ryu.close()
	start_exp = False
	end_exp = False

	self.mac_to_port = {}
        self.topology_api_app = self
        self.datapath_list=[]
	self.datapaths = {}
	self.monitor_thread = hub.spawn(self._monitor)
	self.flowStats = FlowStats()
        #get_data_thread = threading.Thread(target=get_data_socket)	
	#get_data_thread.start()
	#self.fattree = MyTopo()
	get_data_thread = threading.Thread(target=get_data_socket)
	get_data_thread.daemon = True
	get_data_thread.start()
        print 'get_data_thread launched'
	get_topo_thread = threading.Thread(target=get_topo_socket)
	get_topo_thread.daemon = True
	get_topo_thread.start()

        get_exp_done_thread = threading.Thread(target=get_exp_done_socket)
        get_exp_done_thread.daemon = True
        get_exp_done_thread.start()


	print 'get_topo_thread launched'
	runSync_thread = threading.Thread(target=runSync)
	runSync_thread.daemon = True
	runSync_thread.start()
	print 'runSync_thread launched'

	get_rSync_socket_thread = threading.Thread(target=get_rSync_socket)
        get_rSync_socket_thread.daemon = True
        get_rSync_socket_thread.start()

        print 'now send the msg to mininet'
        send_msg_socket()

	global k
	global start_edge
	global end_edge
	global start_agg
	global end_agg
	global start_core
	global end_core
	k = 14
        #k=6 for 1k and 2k vms
        #k=8 for 2k, 3k, and 5k vms
	#k=12 for all
        start_edge = 1
        end_edge = k*(k/2)
        start_agg = end_edge+1
        end_agg = 2*end_edge
        start_core = end_agg+1
        end_core = 2*end_edge+(k/2)**2

	#subprocess.Popen(['/bin/bash', '-c', '>netspeed.dat'])
	#subprocess.Popen(['/bin/bash', '-c', 'sh netspeed.sh eth0 >> netspeed.dat'])

	global nvm
        #global np
        global nf
	global nvpt
        #global server_c
        #global mb_capacity
        #global mb_type
        global fmin
        global fmax

	#nvm = 10000#50000#2000000#5000#10000#100#70 for test
        #np = 10#50
        nf = 100000#100000###3000#50#7 for test
	nvm = 10000#nf//10
	nvpt = 505 # -+25% the approximately number of vm per tenant so size of the group
	#nt = nvm//nvpt#each tenant has at most 50vm - number of tenants or also number of groups

        vm_c = [1] #max 20 vms per host, k = 14 means 432 hosts, server_capacity is 20
        #mb_type = ['FW', 'IPS', 'Proxy']
        #mb_capacity = [500000, 600000, 700000, 800000, 900000, 1000000]#kb/s
        #server_c = 5000000#inf
        fmin = 10#kbit/s
        fmax = 20


    # Handy function that lists all attributes in the given object
    def ls(self,obj):
        print("\n".join([x for x in dir(obj) if x[0] != "_"]))

    
    def add_flow_(self, datapath, in_port, dst, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser      
        match = datapath.ofproto_parser.OFPMatch(in_port=in_port, eth_dst=dst)
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,actions)] 
 
        mod = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=match, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
            priority=ofproto.OFP_DEFAULT_PRIORITY, instructions=inst)
        datapath.send_msg(mod)
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=0, idle_timeout=0, hard_timeout=0, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=0, idle_timeout=0, hard_timeout=0, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)



    def install_path(self, p, ev, src_mac, dst_mac, ip_proto_, ip_src, ip_dst, src_port, dst_port, policy_found,f):
       #print "install_path is called"
       #print mymac
       #print "p=", p, " src_mac=", src_mac, " dst_mac=", dst_mac, " protocol=", ip_proto_, " ipv4_src=", ip_src, " ipv4_dst=",ip_dst, " src_port=", src_port, " dst_port=", dst_port
       #if f not in installed_flows:
         msg = ev.msg
         datapath = msg.datapath
         #if policy_found:
           #print "install_path is called for flow:",f#, str(datapath.id)
	   #print f,ip_src,ip_dst,src_port,dst_port,ip_proto_
         ofproto = datapath.ofproto
         parser = datapath.ofproto_parser
         for sw, in_port, out_port in p:
	   #if policy_found:
             #print f,':',src_mac,"->", dst_mac, "via ", sw, " in_port=", in_port, " out_port=", out_port
	   #if policy_found:
	   if int(ip_proto_)==6: #tcp
             match=parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac, eth_type=0x0800, ip_proto=int(ip_proto_), ipv4_src=ip_src, ipv4_dst=ip_dst, tcp_src=int(src_port), tcp_dst=int(dst_port))
	   elif int(ip_proto_)==17: #udp
	     match=parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac, eth_type=0x0800, ip_proto=int(ip_proto_), ipv4_src=ip_src, ipv4_dst=ip_dst, udp_src=int(src_port), udp_dst=int(dst_port))
	   else:
	     match=parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac, eth_type=0x0800, ip_proto=int(ip_proto_), ipv4_src=ip_src, ipv4_dst=ip_dst)  
	   #else:
	   #  match=parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)	   

           if in_port==out_port:
	     actions=[parser.OFPActionOutput(ofproto.OFPP_IN_PORT)]
	   else:
	     actions=[parser.OFPActionOutput(out_port)]

           datapath=self.datapath_list[int(sw)-1] 
	   #if policy_found:
	     #print 'datapath:',datapath.id

           inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS , actions)]
           mod = datapath.ofproto_parser.OFPFlowMod(
              datapath=datapath, match=match, idle_timeout=0, hard_timeout=0,
              priority=1, instructions=inst)

           datapath.send_msg(mod)
	   installed_flows.append(f)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures , CONFIG_DISPATCHER)
    def switch_features_handler(self , ev):
         print "switch_features_handler is called"
         datapath = ev.msg.datapath
         ofproto = datapath.ofproto
         parser = datapath.ofproto_parser
         match = parser.OFPMatch()
         actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
         inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS , actions)]
         mod = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=match, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
            priority=0, instructions=inst)
         datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
	global f_ryu
	global start_exp
	#if policy_ not in paths_db_mb.keys():
	'''
	if first_packet==False:
	  for k,v in policies.items():
            mb1 = MBList[v[0]][1]
            mb2 = MBList[v[1]][1]
            mb3 = MBList[v[2]][1]
            p2 = get_path(mymac[mb1][0] , mymac[mb2][0], mymac[mb1][1], mymac[mb2][1],False)
            p3 = get_path(mymac[mb2][0], mymac[mb3][0], mymac[mb2][1], mymac[mb3][1],False)
            paths_db_mb[k]=[p2,p3]
            print '--- adding path to paths_db_mb',k
	  first_packet=True
	'''

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
 
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
	icmp_ = pkt.get_protocol(icmp.icmp)
	ipv4_ = pkt.get_protocol(ipv4.ipv4)
	tcp_ = pkt.get_protocol(tcp.tcp)
        udp_ = pkt.get_protocol(udp.udp)
	#for p in pkt:
	#    if p.protocol_name not in ['lldp']:
	    #if p.protocol_name == 'ipv4':
	#	print p.protocol_name#, p
		#ip = p.get_protocols(ipv4.ipv4)
		#print p.src, p.dst
	    #print "Protocol: ", p
        #print "eth.ethertype=", eth.ethertype
 
        #avodi broadcast from LLDP
        if eth.ethertype==35020:
          return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
	if icmp_:
	    code = icmp_.code
	else:
	    code = 'NA'
	if ipv4_:
	    ip_src = ipv4_.src
	    ip_dst= ipv4_.dst
	    ip_proto_ = ipv4_.proto
	else:
	    ip_src = 'NA'
	    ip_dst = 'NA'
	if tcp_:
	    tcp_port_src = tcp_.src_port
	    tcp_port_dst = tcp_.dst_port
	else:
	    tcp_port_src = 'NA'
	    tcp_port_dst = 'NA'
        if udp_:
            udp_port_src = udp_.src_port
            udp_port_dst = udp_.dst_port
        else:
            udp_port_src = 'NA'
            udp_port_dst = 'NA'

	#list_proto = []
	#for p in pkt:
	    #print p.protocol_name
	#print "protocols: ", list_proto
	#print "from ethernet: ",dst,src,dpid," from icmp: ",code," from ipv4",src_ip,dst_ip,ip_proto_," from tcp: ",src_port, dst_port
        self.mac_to_port.setdefault(dpid, {})

 	#update_mymac_mbs()
	policy_found = None
	f = None
        if src not in mymac.keys():
            mymac[src]=( dpid,  in_port)
            #print "mymac=", mymac
	    #print 'mymac_mb',mymac_mb
 
        if dst in mymac.keys():
	    #print "mymac=", mymac
	    #print 'mymac_mb=',dst,mymac_mb[dst],mymac[dst]
	    #msg = None
            #get policy id by identifying src & dst (to add port and proto), (change header?)
	    #example src=00:00:00:00:00:01 dst=00:00:00:00:00:08 
	    #should ping all mbs to fill up mymac dict, a fix will come up soon
	    #print 'start searching policy for:',ip_src,' in ', get_host_ip(ip_src), ip_dst, ' in ',get_host_ip(ip_dst), tcp_port_src, tcp_port_dst, udp_port_src, udp_port_dst, ip_proto_
	    #print ip_src, ip_dst, tcp_port_src, tcp_port_dst, ip_proto_
	    if int(ip_proto_)==6:
	      p,mb_list,f = get_policy(ip_src, ip_dst, tcp_port_src, tcp_port_dst, ip_proto_)
	    elif int(ip_proto_)==17:
	      p,mb_list,f = get_policy(ip_src, ip_dst, udp_port_src, udp_port_dst, ip_proto_)
	    else: 
	      p,mb_list,f = get_policy(ip_src, ip_dst, 'NA', 'NA', ip_proto_)
	    #print 'found:',p,mb_list
	    policy_=p
	    if len(mb_list) > 0: # there is a policy
		mb1 = mb_list[0]
		mb2 = mb_list[1]
		mb3 = mb_list[2]
		#print 'found:',f,p,mb1,get_mb_s_t(mb1), mb2,get_mb_s_t(mb2), mb3, get_mb_s_t(mb3)
	
		#print 'packet received by the controller from',dpid,f,p
		#policy_found = True
		#print "found policy:", p, "for:",ip_src,' in ', get_host_ip(ip_src), ip_dst, ' in ',get_host_ip(ip_dst), tcp_port_src, tcp_port_dst, udp_port_src, udp_port_dst, ip_proto_ 
		#mb1, get_mb_s_t(mb1), mb2,get_mb_s_t(mb2), mb3, get_mb_s_t(mb3)
		#print mymac
		#print "traffic goes from ", src, " to ", dst, " through mb1:",get_mb_s_t(mb1)," mb2:",get_mb_s_t(mb2)," mb3:",get_mb_s_t(mb3)
		
		f_ryu_name = 'time_ryu.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'.dat'
		
		global f_ryu_vmstat_name
		#global ryu_vmstat
		f_ryu_vmstat_name = 'vmstat_ryu.'+str(len(flows)/2)+'_'+str(len(policies)/2)+'.dat'
	        f_ryu=None	
		if start_exp == False:
		  for k,v in policies.items():
                    mb1_ = MBList[v[0]][1]
                    mb2_ = MBList[v[1]][1]
                    mb3_ = MBList[v[2]][1]
                    p2 = get_path(mymac[mb1_][0] , mymac[mb2_][0], mymac[mb1_][1], mymac[mb2_][1],False)
                    p3 = get_path(mymac[mb2_][0], mymac[mb3_][0], mymac[mb2_][1], mymac[mb3_][1],False)
                    paths_db_mb[k]=[p2,p3]
                    print '--- adding path to paths_db_mb',k

		  start_exp = True
		  f_ryu=open(f_ryu_name, 'w')

		  f_ryu_vmstat = open(f_ryu_vmstat_name, 'w')
		  f_ryu_vmstat.close()

		  ryu_vmstat = subprocess.Popen(['sh', 'vmstat.sh'] + [f_ryu_vmstat_name]) 
		else:
		  f_ryu=open(f_ryu_name, 'a+')
		
		start_time = time.time()		
		
		#if (policy_,src,dst) not in paths_db.keys():
		p1 = get_path(mymac[src][0], mymac[mb1][0], mymac[src][1], mymac[mb1][1],policy_found)

		'''
		if policy_ not in paths_db_mb.keys():
		  p2 = get_path(mymac[mb1][0] , mymac[mb2][0], mymac[mb1][1], mymac[mb2][1],policy_found)
		  p3 = get_path(mymac[mb2][0], mymac[mb3][0], mymac[mb2][1], mymac[mb3][1],policy_found)
		  paths_db_mb[policy_]=[p2,p3]
		  print '--- adding path to paths_db_mb',policy_
		else:
		  #print len(paths_db_mb),policy_
		  p2=paths_db_mb[policy_][0]
		  p3=paths_db_mb[policy_][1]
		  #print policy_,p2+p3
		'''

		p2=paths_db_mb[policy_][0]
                p3=paths_db_mb[policy_][1]

		policy_found = True
                p4 = get_path(mymac[mb3][0], mymac[dst][0], mymac[mb3][1], mymac[dst][1],policy_found)
		  #print "p1",p1
		  #print "p2",p2
		  #print "p3",p3
		  #print "p4",p4
		  #print 'original path before cleaning up:',p1+p2+p3+p4
		p_ = [x[0] for x in groupby(p1+p2+p3+p4)] 
		  #p = clean_path(clean_path(p_))
		p = clean_path(p_)
		#print p
		#paths_db[(policy_,src,dst)]=p
		#print 'adding path to paths_db - size is',len(paths_db)
		#else:
		#p = paths_db[(policy_,src,dst)]
		
		#start_time = time.time() 
		if int(ip_proto_)==6:
                  self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst, tcp_port_src, tcp_port_dst,policy_found,f)
		elif int(ip_proto_)==17:
		  self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst, udp_port_src, udp_port_dst,policy_found,f)
		else:
		  self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst, 'NA', 'NA',policy_found,f)
                out_port = p[0][2]
		
		r_msg = '%s,%s,%s,%s' % (len(flows)/2,dpid,f,time.time() - start_time)
		print r_msg#,policy_
		f_ryu.write(r_msg+"\n")
		#f_ryu.close()

	    else:
		policy_found = False
		#print mymac[src][0], mymac[dst][0], mymac[src][1], mymac[dst][1]
	        p = get_path(mymac[src][0], mymac[dst][0], mymac[src][1], mymac[dst][1],policy_found)
		#print dpid,p
	        #print "datapath id: ", dpid, " / path is", p
		
		if int(ip_proto_)==6:
	          self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst, tcp_port_src, tcp_port_dst,policy_found,f)
		elif int(ip_proto_)==17:
		  self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst, udp_port_src, udp_port_dst,policy_found,f)
		else:
		  self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst, 'NA', 'NA',policy_found,f)
		#self.install_path(p, ev, src, dst, 'NA','NA','NA', 'NA', 'NA',policy_found,f)
		#self.install_path(p, ev, src, dst, ip_proto_, ip_src, ip_dst,'NA', 'NA',policy_found,f)
	        out_port = p[0][2]
        else:
            out_port = ofproto.OFPP_FLOOD
       
        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
	    #if policy_found:
	      if int(ip_proto_)==6: #tcp 
                match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst, eth_type=0x0800, ip_proto=int(ip_proto_), ipv4_src=ip_src, ipv4_dst=ip_dst, tcp_src=int(tcp_port_src), tcp_dst=int(tcp_port_dst))
	      elif int(ip_proto_)==17:#udp
	    	match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst, eth_type=0x0800, ip_proto=int(ip_proto_), ipv4_src=ip_src, ipv4_dst=ip_dst, udp_src=int(udp_port_src), udp_dst=int(udp_port_dst))
	      else: #not tcp, ipv4
	    	match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst, eth_type=0x0800, ip_proto=int(ip_proto_), ipv4_src=ip_src, ipv4_dst=ip_dst)
	    #else:
	    #  match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)

	    #if msg.buffer_id!=ofproto.OFP_NO_BUFFER:
	    #	self.add_flow(datapath, 1, match, actions, policy_found,f,msg.buffer_id)
	    #	return
	    #else:
	    #	self.add_flow(datapath, 1, match, actions,policy_found,f) 
	    

        data=None
        if msg.buffer_id==ofproto.OFP_NO_BUFFER:
           data=msg.data
 
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
            actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):
        global switches
        global k
        global start_edge
        global end_edge
        global start_agg
        global end_agg
        global start_core
        global end_core

        switch_list = get_switch(self.topology_api_app, None)  
        switches=[switch.dp.id for switch in switch_list]
        self.datapath_list=[switch.dp for switch in switch_list]
        #print "self.datapath_list=", self.datapath_list
        #print "switches=", switches

        links_list = get_link(self.topology_api_app, None)
        mylinks=[(link.src.dpid,link.dst.dpid,link.src.port_no,link.dst.port_no) for link in links_list]
        for s1,s2,port1,port2 in mylinks:
          adjacency[s1][s2]=port1
          adjacency[s2][s1]=port2
	  #check type of s1 and s2 (edge,agg,core)
	  # it depends on k
	  # edge = range(0,k*(k/2)); agg = range(0,k*(k/2)); core=range(0,(k/2)**2)
	  #end_edge = k*(k/2)
	  #start_agg = end_edge+1
	  #end_agg = start_agg+end_edge
	  #print s1,s2
	  if (int(s1) in range(int(start_edge),int(end_edge)+1) and int(s2) in range(int(start_agg),int(end_agg)+1)) or (int(s2) in range(int(start_edge),int(end_edge)+1) and int(s1) in range(int(start_agg),int(end_agg)+1)):
            weight[(s1,s2)]=3
	    weight[(s2,s1)]=3
	  elif (int(s1) in range(int(start_agg), int(end_agg)+1) and int(s2) in range(int(start_core), int(end_core)+1)) or (int(s2) in range(int(start_agg), int(end_agg)+1) and int(s1) in range(int(start_core), int(end_core)+1)):
	    weight[(s1,s2)]=5
            weight[(s2,s1)]=5#to be updated
          #print "port:",s1,s2,port1,port2
	  #print "weight:",s1,s2,weight[(s1,s2)],weight[(s2,s1)]
	#print len(switches),end_core
	if len(switches)==int(end_core):
	  print '******* all switched in the topology *******'
	'''
	if len(switches)==int(end_core):
	  #global paths_db_mb
          policy_found=False
	  #print len(policies)
          for k,v in policies.items():
            mb1 = MBList[v[0]][1]
            mb2 = MBList[v[1]][1]
            mb3 = MBList[v[2]][1]
            #print k,mb1,mb2,mb3
            #print policies
            p2=None
            p3=None
            p2 = get_path(mymac[mb1][0] , mymac[mb2][0], mymac[mb1][1], mymac[mb2][1],policy_found)
            p3 = get_path(mymac[mb2][0], mymac[mb3][0], mymac[mb2][1], mymac[mb3][1],policy_found)
            paths_db_mb[k]=[p2,p3]
	    print v[0],v[1],v[2]
	    print p2+p3
            print 'path added to policy', k
	'''

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
	#global start_exp
	#global end_exp
	#global f_ryu
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
	    #if start_exp == False:
	    #    f_ryu=open('ryu_'+str(len(flows)/2)+'.csv', 'a+')
	    #    start_exp = True
	    #	end_exp = False
            if not datapath.id in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
		print 'register datapath: ', datapath.id
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
	    #if end_exp == False and start_exp == True:
	    #	f_ryu.close()
	    #	end_exp = True
	    #	start_exp = False
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
		print 'unregister datapath: ', datapath.id
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            self.flowid = 0
            self.flowStats.emptyFlows()
            for dp in self.datapaths.values():
                if 1 <= dp.id <= 8: #edge switches
                    self._request_stats(dp)
		    #print "len of MBList", len(MBList)
            hub.sleep(5)
            #self.flowStats.printFlows()
            #self.flowStats.getNCVG()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)


    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body

#        self.logger.debug('datapath         '
#                         'in-port  eth-dst           '
#                         'out-port packets  bytes')
#        self.logger.debug('---------------- '
#                         '-------- ----------------- '
#                         '-------- -------- --------')
        for stat in sorted([flow for flow in body if flow.priority == 1 and flow.match['in_port'] in [1, 2]],#only outgoing traffic
                           key=lambda flow: (flow.match['in_port'],
                                             flow.match['eth_dst'])):

#            self.logger.debug('%016x %8x %17s %8x %8d %8d',
#                             ev.msg.datapath.id,
#                             stat.match['in_port'], stat.match['eth_dst'],
#                             stat.instructions[0].actions[0].port,
#                             stat.packet_count, stat.byte_count)
	    '''
            if stat.match['in_port'] == 1: #to identify the src mac address
                if ev.msg.datapath.id == 1:
                    eth_src = '00:00:00:00:00:01'
                elif ev.msg.datapath.id == 2:
                    eth_src = '00:00:00:00:00:03'
            elif stat.match['in_port'] == 2:
                if ev.msg.datapath.id == 1:
                    eth_src = '00:00:00:00:00:02'
                elif ev.msg.datapath.id == 2:
                    eth_src = '00:00:00:00:00:04'
	    '''
	    eth_src = get_mac(ev.msg.datapath.id, stat.match['in_port'])
            self.flowStats.addFlow([self.flowid, ev.msg.datapath.id,
                             stat.match['in_port'], eth_src, stat.match['eth_dst'],
                             stat.instructions[0].actions[0].port,
                             stat.packet_count, stat.byte_count])
            self.flowid += 1


