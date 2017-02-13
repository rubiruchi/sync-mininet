import itertools

class FlowStats(object):
    def __init__(self, *args, **kwargs):
        super(FlowStats, self).__init__(*args, **kwargs)
        self.Flows = []
        self.Groups = []
        self.ActiveVms = []


    def addFlow(self, flow):
        self.Flows.append(flow)

    def emptyFlows(self):
        del self.Flows[:]

    def getVmFlows(self, vm):
        k = []
        for f in self.Flows:
            if f[3] == vm or f[4] == vm:
                k.append([f[0],f[3],f[4]])

        k.sort()
        VmFlows = list(k for k,_ in itertools.groupby(k))
        return VmFlows

    def getActiveVms(self):
        del self.ActiveVms[:]
        for f in self.Flows:
            self.ActiveVms.append(f[3])
            self.ActiveVms.append(f[4])

        self.ActiveVms = list(set(self.ActiveVms))

    def printActiveVms(self):
        print 'Printing Active Vms...'
        for f in self.ActiveVms:
            print f,
        print

    def printFlows(self):
	if len(self.Flows) > 0:
            self.Flows = sorted(self.Flows, key = lambda x: (x[0], x[1]))
            print 'Printing flows...'
            print 'flow-id   datapath  in-port   eth-src            eth-dst            out-port  packets   bytes'
            print '--------  --------  --------  -----------------  -----------------  --------  --------  --------'
            for f in self.Flows:
                print '%8s  %8s  %8s  %17s  %17s  %8s  %8s  %8s' % (f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7])
	#else:
	    #print 'No flows to print...'

    def getNCVG(self):

        del self.Groups[:]

        self.getActiveVms()

        while len(self.ActiveVms) > 0:
            Group = []
            Group.append(self.ActiveVms[0])
            VmFlows = self.getVmFlows(self.ActiveVms[0])
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
                k = VmFlows + self.getVmFlows(vp)
                k.sort()
                VmFlows = list(k for k,_ in itertools.groupby(k))
                VmFlows.remove(fj)

            self.Groups.append(Group)
            self.ActiveVms = [vm for vm in self.ActiveVms if vm not in Group]
	if len(self.Groups) > 0:
            print 'Printing Vm Groups...'
            for g in self.Groups:
                print g
