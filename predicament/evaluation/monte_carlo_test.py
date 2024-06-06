import numpy as np


def monte_carlo_test_confusion_matrix(cm, scoring='accuracy', trials=100000):
    """
    This isn't quite a permutation test but it is close. It estimates with what
    probability would I see the same accuracy of cm by chance (where by-chance
    is defined in terms of our null hypothesis conditions).
    
    It says for some confusion matrix cm, the null hypothesis is that each
    row is sampled from a multinomial with probabilities equal to the 
    normalised column sums.
    
    E.g. for cm = [[10,0,0],[5,5,0],[4,0,4]]
    Column sums are [19,5,4] total sum is 28 so null hyp dist is [19/28,5/28,4/28]
    
    
    In fact, I have added a bayesian approximation for this. A randomly sampled 
    cm is one where each row r was n_r samples from the multinomial
    where n_r is the row sum.
    
    Then the monte_carlo test (monte carlo test) says with what probability would
    a randomly sampled cm have the same or larger diagonal sum than the
    input cm
    
    """
    score = score_from_cm(cm, scoring)
    geq_cout = 0
    for i in range(trials):
        sampled_cm = sample_cm(cm)
        sample_score = score_from_cm(sampled_cm, scoring)
        if sample_score >= score:
            geq_cout += 1
    return geq_cout/trials

def sample_cm(cm, alpha=0):
    pvals = cm.sum(axis=0)+alpha
    pvals = pvals/np.sum(pvals)
    sample = np.array(
        [ np.random.multinomial(n,pvals) for n in cm.sum(axis=1) ])
    return sample

def score_from_cm(cm, scoring):
    if scoring == 'accuracy':
        return np.sum(cm.diagonal())/cm.sum()
    else:
        raise ValueError(f"Unrecognised scoring {scoring}")



