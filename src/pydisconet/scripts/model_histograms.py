
def histogram(data_object=None,z=None, save='histogram.png',pred=None,label=None):
    if pred is None:
        if data_object is not None:
            data_object= data_object.to('cpu')
            if z is None:
                z = data_object.x
            else:
                z = z.detach().to('cpu')

            edge_lbl = torch.clamp(data_object.edge_label, max=1)
            pos_edge_index = data_object.edge_label_index[:,edge_lbl==1]
            neg_edge_index = data_object.edge_label_index[:,edge_lbl==0]
            pos_pred = torch.clamp((F.cosine_similarity(z[pos_edge_index[0]], z[pos_edge_index[1]], dim=1)*0.5)+0.5,min=0,max=1)
            neg_pred = torch.clamp((F.cosine_similarity(z[neg_edge_index[0]], z[neg_edge_index[1]], dim=1)*0.5)+0.5,min=0,max=1)
            pred = torch.cat([pos_pred, neg_pred], dim=0)
        else:
            raise ValueError("Data Object and pred cannot be None at the same time. Provide one of them.")
    else:
        if label is not None:
            label = label.to('cpu')
            pos_pred = pred[label==1]
            neg_pred = pred[label==0]
        else:
            raise ValueError("Labels are required for pred")

    # Plot Histogram of Predicted Values with different colors
    plt.figure(figsize=(8, 4))
    plt.hist(pos_pred, bins=50, color='green', alpha=0.7, label='Positive Labels')
    plt.hist(neg_pred, bins=50, color='red', alpha=0.7, label='Negative Labels')
    # Different criterias
    plt.axvline(x=(pos_pred.mean()+ neg_pred.mean())/2 , color='b', linestyle='--',label='mean_label')
    plt.axvline(x=(pred).mean().detach().cpu(), color='r', linestyle='--',label='mean_dotpro')
    
    plt.xlabel('Dot Product')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dot Product for different labels')
    plt.legend()
    plt.savefig(f"{save}")
    
    return None
