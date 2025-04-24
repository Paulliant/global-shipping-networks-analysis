import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from models import StructureEncoder

class Trainer():
	def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn

		# 结构信息和其编码网络
		self.struct_feats_cache = {}
		self.encoder = StructureEncoder(input_dim=len(self.args.structure_feats), output_dim=self.args.transform_layer_feats).to(self.args.device)

		self.classifier = classifier
		self.comp_loss = comp_loss

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)

		if self.tasker.is_static:
			adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
			self.hist_adj_list = [adj_matrix]
			self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

	def init_optimizers(self,args):
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()

	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	def train(self):
		self.tr_step = 0
		best_eval_valid = 0
		eval_valid = 0
		epochs_without_impr = 0

		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
				if eval_valid>best_eval_valid:
					best_eval_valid = eval_valid
					epochs_without_impr = 0
					print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
				else:
					epochs_without_impr+=1
					if epochs_without_impr>self.args.early_stop_patience:
						print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
						break

			if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

				if self.args.save_node_embeddings and nodes_embs is not None:
					if self.tasker.is_static:
						self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
						self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
						self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')
					else:
						self.save_node_embs_csv_dynamic(nodes_embs, range(self.data.num_nodes), f'embedding/nodeembs_epoch{e}.csv')

	def get_l1_loss(self, nodes_embs, path):
		# 读取 CSV，按第一列（节点 ID）排序，去掉第一列
		ref_embs = pd.read_csv(path, header=None).sort_values(by=0).iloc[:, 1:].values
		ref_embs = torch.tensor(ref_embs, dtype=torch.float32, device=nodes_embs.device)

		# L1 loss
		return torch.nn.functional.l1_loss(nodes_embs, ref_embs)

	def run_epoch(self, split, epoch, set_name, grad):
		t0 = time.time()
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

		torch.set_grad_enabled(grad)
		for idx, s in enumerate(split):
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)

			predictions, nodes_embs = self.predict(
												s.hist_ori_adj_list,
												s.hist_adj_list,
												s.hist_ndFeats_list,
												s.label_sp['idx'],
												s.node_mask_list, 
												idx)	# split的编号

			loss = self.comp_loss(predictions,s.label_sp['vals'])
			if hasattr(self.args, 'l1_loss_mode') and self.args.l1_loss_mode:
				loss += self.args.l1_loss_weight * self.get_l1_loss(nodes_embs, self.args.l1_loss_file)
			# print(loss)

			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			if grad:
				self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done()

		return eval_measure, nodes_embs

	# 计算结构信息的函数
	def compute_node_metrics(self, hist_adj_list):
		# 聚合所有历史邻接矩阵中的边
		edge_weights = {}
		for adj in hist_adj_list:
			adj = adj.coalesce()
			indices = adj.indices().cpu().numpy()
			values = adj.values().cpu().numpy()
			for u, v, w in zip(indices[0], indices[1], values):
				key = (int(u), int(v))
				if key not in edge_weights:
					edge_weights[key] = 0.0
				edge_weights[key] += float(w)
		
		# 转为 DataFrame
		edges = list(edge_weights.keys())
		weights = list(edge_weights.values())
		df = pd.DataFrame(edges, columns=['source', 'target'])
		# df['weight'] = weights  # 用真实累加权重填充
		df['weight'] = 1.0

		# 构造有向图
		G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

		nodes = list(range(0, self.data.num_nodes))

		degree = dict(nx.degree_centrality(G))
		betweenness = dict(nx.betweenness_centrality(G))
		closeness = dict(nx.closeness_centrality(G))
		pagerank = dict(nx.pagerank(G))
		
		df_metrics = pd.DataFrame({
			'label': nodes,
			'degree': [degree.get(n, 0.0) for n in nodes],
			'betweenness': [betweenness.get(n, 0.0) for n in nodes],
			'closeness': [closeness.get(n, 0.0) for n in nodes],
			'pagerank': [pagerank.get(n, 0.0) for n in nodes]
		})

		df_metrics = df_metrics.set_index('label').sort_index()
		
		return df_metrics

	def predict(self,hist_ori_adj_list,hist_adj_list,hist_ndFeats_list,node_indices,mask_list,idx):
		nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)
		
		if hasattr(self.args, 'structure_feats_mode') and self.args.structure_feats_mode != 'normal':
			
			if idx not in self.struct_feats_cache:
				# 获取静态图的结构特征
				struct_df = self.compute_node_metrics(hist_ori_adj_list)
				struct_df = struct_df[self.args.structure_feats]
				struct_feats = torch.tensor(struct_df.values, dtype=torch.float32).to(self.args.device)
				self.struct_feats_cache[idx] = struct_feats

			# 通过mlp
			struct_feats_encoded = self.encoder(self.struct_feats_cache[idx])

			# 只是用结构特征
			if self.args.structure_feats_mode == 'structure_only':
				nodes_embs = struct_feats_encoded
			# 使用嵌入特征和结构特征
			elif self.args.structure_feats_mode == 'structure_added':
				nodes_embs = torch.cat([nodes_embs, struct_feats_encoded], dim=1)

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	def optim_step(self,loss):
		self.tr_step += 1
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.step()
			self.classifier_opt.step()

			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()


	def prepare_sample(self,sample):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		for i, adj in enumerate(sample.hist_ori_adj_list):	# 将原始的边也转移到设备上
			adj = u.sparse_prepare_tensor(adj, torch_size=[self.num_nodes])
			sample.hist_ori_adj_list[i] = adj.to(self.args.device)

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	def prepare_static_sample(self,sample):
		sample = u.Namespace(sample)

		sample.hist_adj_list = self.hist_adj_list

		sample.hist_ndFeats_list = self.hist_ndFeats_list

		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp

		return sample
		

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)
	
	def save_node_embs_csv_dynamic(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			row = torch.cat((torch.tensor([node_id]), nodes_embs[node_id].double().cpu())).detach().numpy()
			csv_node_embs.append(row)

		df = pd.DataFrame(np.array(csv_node_embs))

		df.to_csv(file_name, header=None, index=None)