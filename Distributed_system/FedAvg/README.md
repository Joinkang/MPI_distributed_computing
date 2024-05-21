I wanna help you guys to use mpi4py in nvidia nano boards for distibuted system

I used the commands ' mpiexec  --allow-run-as-root -bind-to none --mca btl_tcp_if_include eth0 --map-by node -np 1 --hostfile my_host_server python3 {server.py} : -np 15 --hostfile my_host_server python3 {clients.py}'

This command use the 1 server and 14 clients node
