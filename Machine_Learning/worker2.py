# -*- coding: utf-8 -*-

import tensorflow as tf 
worker1 = "121.248.96.53:2222"
worker2 = "121.248.96.97:2223"
worker_hosts = [worker1, worker2]
cluster_spec = tf.train.ClusterSpec({ "worker": worker_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
server.join()
