import os
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F




def enable_dropout(model):#enable dropout for monte carlo dropout
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def calculate_uncertainty(inputs,model):
    outputs = model(inputs)
    return torch.abs(torch.softmax(outputs, dim=1)[:,1] - 0.5) 


def monte_carlo_uncertainty(inputs,model):

    n_samples=25
    batch_size, *_ = inputs.shape
    predictions = torch.zeros((n_samples, batch_size))

    for i in range(n_samples):
        predictions[i] =(torch.softmax( model(inputs), dim=1))[:,1]
    
    uncertainty = predictions.std(0)
    return  uncertainty



def select_uncertain(params, unl_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=500):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    params.model = params.model.to(device)
    params.model.eval()
    uncertainties = []
    all_indices = [j for j in range(0, unl_dataloader.dataset.__len__())]
    progress_bar = tqdm(unl_dataloader)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            uncertainty = calculate_uncertainty(inputs,params.model)
            uncertainties.extend(uncertainty.cpu().numpy())

    # Sort samples by uncertainty
    indices_uncertainties = list(zip(all_indices, uncertainties))
    indices_uncertainties.sort(key=lambda x: x[1])  

    
    # Split the dataset
    uncertain_indices = [idx for idx, _ in indices_uncertainties[:split_size]]


    
    return uncertain_indices

def select_uncertain_carlo(params, unl_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=500):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    params.model = params.model.to(device)
    params.model.eval()
    uncertainties = []
    all_indices = [j for j in range(0, unl_dataloader.dataset.__len__())]
    progress_bar = tqdm(unl_dataloader)
    enable_dropout(params.model)#for monte carlo dropout
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            uncertainty=monte_carlo_uncertainty(inputs,params.model)
            uncertainties.extend(uncertainty.cpu().numpy())

    # Sort samples by uncertainty
    indices_uncertainties = list(zip(all_indices, uncertainties))
    indices_uncertainties.sort(key=lambda x: x[1],reverse=True)
    
    # Split the dataset
    uncertain_indices = [idx for idx, _ in indices_uncertainties[:split_size]]

    return uncertain_indices

def select_random(len,size=500):
    indexes = np.random.choice(range(len), size=size, replace=False)
    return indexes



def get_vectors(params, dataloader, device,checkpoints_dir_name):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    model=params.model.eval()
    model.classifier = torch.nn.Identity()
    progress_bar = tqdm(dataloader)
    progress_bar.set_description("get vectors")
    model.to(device)
    vectors=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            outputs = model(inputs)
            vectors.extend(outputs)

        
    return vectors

def get_uncertainty(params, unl_dataloader, device,
                     checkpoints_dir_name):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    params.model = params.model.to(device)
    params.model.eval()
    uncertainties = []
    progress_bar = tqdm(unl_dataloader)
    progress_bar.set_description("get uncertainty")
    #enable_dropout(params.model)#for monte carlo dropout
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            #uncertainty=monte_carlo_uncertainty(inputs,params.model)
            uncertainty = calculate_uncertainty(inputs,params.model)
            uncertainties.extend(uncertainty)

    
    return uncertainties




def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        #np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            if i % print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]

def metric(batch_i,batch_j):
    unci=batch_i[:,-1:]
    uncj=batch_j[:,-1:]
    batch_i=batch_i[:,:-1]
    batch_j=batch_j[:,:-1]
    inorm = F.normalize(batch_i, p=2, dim=1)
    jnorm = F.normalize(batch_j, p=2, dim=1)
    uncertainties =0.5- torch.max(unci, uncj.t())

    dis =1- torch.mm(inorm, jnorm.t())
    return dis*0.6+0.4*uncertainties


def div_unc(params, unl_dataloader, device,
                    tb_dir_name, checkpoints_dir_name,split_size=500):
    uncertainties=get_uncertainty(params, unl_dataloader, device,checkpoints_dir_name)
    
    vectors=get_vectors(params, unl_dataloader, device,checkpoints_dir_name)

    uncertainties = torch.stack(uncertainties).unsqueeze(1)
    vectors = torch.stack(vectors)
    print(uncertainties.shape)
    print(vectors.shape)
    list=torch.cat((vectors, uncertainties),1)
    print(list[0].shape)
    zeros_list = [0 for _ in range(unl_dataloader.dataset.__len__())]
    indices=k_center_greedy(list, split_size, metric, device,already_selected=[])

    """
    num_items=unl_dataloader.dataset.__len__()
    L = np.zeros((num_items,num_items ))
    logging.info("calculate matrix")
    vectors = torch.stack(vectors).to(device)
    uncertainties = torch.stack(uncertainties).to(device)
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    normalized_vectors = vectors / norms
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t()).cpu().numpy()
    uncertainty_matrix = torch.max(uncertainties.unsqueeze(1), uncertainties.unsqueeze(0)).cpu().numpy()
    L = 0.4 * uncertainty_matrix + 0.6 * similarity_matrix
    logging.info("DPP")
    dpp = FiniteDPP('likelihood', L=L)
    rng = RandomState(123)
    # Sample x diverse items
    dpp.sample_mcmc_k_dpp(size=size, random_state=rng)  
    #dpp.sample_exact_k_dpp(size=size)
    sampled_indices = dpp.list_of_samples[-1]
    """
    
    return indices

def disagreement():
    """could implement a treshold or treshold + must be at the other side of the 0.5 mark"""
    return None


def select_disagreement(params,params_aux, unlabelled_dataloader, device,
                        tb_dir_name, checkpoints_dir_name,size=500):
    state = torch.load(os.path.join(checkpoints_dir_name,'primary_best.pt'))
    params.model.load_state_dict(state['net'])
    model=params.model.eval()
    model=model.to(device)
    state = torch.load(os.path.join(checkpoints_dir_name,'auxiliary_best.pt'))
    params_aux.model.load_state_dict(state['net'])
    aux=params_aux.model.eval()
    aux=aux.to(device)
    progress_bar = tqdm(unlabelled_dataloader)
    progress_bar.set_description("Looking for disagreements")
    all_indices = [j for j in range(unlabelled_dataloader.dataset.__len__())]
    disagreements=[]
    with torch.no_grad():
        for _, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            output1 = model(inputs)
            output2 = aux(inputs)
            dis=torch.abs(torch.softmax(output1, dim=1)[:,1]-torch.softmax(output2, dim=1)[:,1])#higher value =higher disagreement
            disagreements.extend(dis.cpu().numpy())
    indices_disagreement = list(zip(all_indices, disagreements))
    indices_disagreement.sort(key=lambda x: x[1],reverse=True)
        
    print(indices_disagreement[0][1])
    return [idx for idx, _ in indices_disagreement[:size]]