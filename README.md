There are a lot of code commented/unused in fattree topology class as it has been moved to the controller.
The code needs to be documented and cleaned up
Some sh files in the controller are used to get statistics of CPU/Memory usage and of other metrics

Sockets is the mean of communication between mininet and the controller, to make this possible, you should add these forwarding rules on the two servers:

On mininet:
ssh -L mininet-host:50009:127.0.0.1:50009 user@mininet-host
ssh -L mininet-host:50010:127.0.0.1:50010 user@mininet-host
ssh -L mininet-host:50012:127.0.0.1:50012 user@mininet-host
To run the topology:
sudo python fattree_topo.py

On the controller:
ssh -L controller-host:50011:127.0.0.1:50011 user@controller-host
ssh -L controller-host:50008:127.0.0.1:50008 user@controller-host
ssh -L controller-host:50007:127.0.0.1:50007 user@controller-host
ssh -L controller-host:50013:127.0.0.1:50013 user@controller-host
To run the controller:
sudo ryu-manager --observe-links sync_routing.py
