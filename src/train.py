from torch import nn
from torch.nn import MSELoss
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from Module.XLNet_LFusion import XLNetForSequenceClassification
from src.eval_metrics import *
from src.utils import *


def initiate(hyp_params, num_train_optimization_steps):
    model = XLNetForSequenceClassification.from_pretrained(hyp_params.model, num_labels=1)
    model.to(hyp_params.device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # Prepare optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=hyp_params.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hyp_params.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    settings = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
    return settings


def train_model(settings, hyp_params, train_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    scheduler = settings['scheduler']
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_loader, desc="Train Iteration")):
        batch = tuple(t.to(hyp_params.device) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        logits = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask)
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))  # 计算最终的损失

        if hyp_params.gradient_accumulation_step > 1:
            loss = loss / hyp_params.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % hyp_params.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def evaluate_model(settings, hyp_params, valid_loader):
    model = settings['model']
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_loader, desc="Valid Iteration")):  # 加载DataLoader
            batch = tuple(t.to(hyp_params.device) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            logits = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask)
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if hyp_params.gradient_accumulation_step > 1:
                loss = loss / hyp_params.gradient_accumulation_step
            dev_loss += loss.item()
            nb_dev_steps += 1
    return dev_loss / nb_dev_steps


def test_model(model, hyp_params, test_loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Iteration"):
            batch = tuple(t.to(hyp_params.device) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            logits = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
        preds = np.array(preds)
        labels = np.array(labels)
    return preds, labels


def train(hyp_params, train_loader, valid_loader, test_loader, num_train_optimization_steps):
    settings = initiate(hyp_params, num_train_optimization_steps)
    test_mae = []
    test_corr = []
    test_f_score = []
    test_accuracies = []
    for epoch in range(1, hyp_params.n_epochs + 1):
        train_loss = train_model(settings, hyp_params, train_loader)
        valid_loss = evaluate_model(settings, hyp_params, valid_loader)
        print("\n" + "-" * 50)
        print('Epoch {:2d}| Train Loss {:5.4f} | Valid Loss {:5.4f} '.format(epoch, train_loss, valid_loss))
        print("-" * 50)
        results, truths = test_model(settings['model'], hyp_params, test_loader)
        acc, mae, corr, f_score = test_score_model(results, truths)
        test_mae.append(mae)
        test_corr.append(corr)
        test_f_score.append(f_score)
        test_accuracies.append(acc)
    print('The test results:\n| MAE {:5.4f} | Corr {:5.4f} | F1 {:5.4f} | ACC {:5.4f}'.format(min(test_mae),
                                                                                              max(test_corr),
                                                                                              max(test_f_score),
                                                                                              max(test_accuracies)))
