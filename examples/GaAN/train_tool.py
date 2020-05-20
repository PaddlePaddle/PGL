import time
from pgl.utils.logger import log

def train_epoch(batch_iter, exe, program, loss, score, evaluator, epoch, log_per_step=1):
    batch = 0
    total_loss = 0.0
    total_sample = 0
    result = 0
    for batch_feed_dict in batch_iter():
        batch += 1
        batch_loss, y_pred = exe.run(program, fetch_list=[loss, score], feed=batch_feed_dict)
        
        num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
        total_sample += num_samples
        input_dict = {
            "y_true": batch_feed_dict["node_label"],
#             "y_pred": y_pred[batch_feed_dict["node_index"]]
            "y_pred": y_pred
        }
        result += evaluator.eval(input_dict)["rocauc"]

#         if batch % log_per_step == 0:
#             print("Batch {}: Loss={}".format(batch, batch_loss))
#             log.info("Batch %s %s-Loss %s %s-Acc %s" %
#                      (batch, prefix, batch_loss, prefix, batch_acc))
        
#     print("Epoch {} Train: Loss={}, rocauc={}, Speed(per batch)={}".format(
#         epoch, total_loss/total_sample, result/batch, (end-start)/batch))
    return total_loss.item()/total_sample, result/batch

def inference(batch_iter, exe, program, loss, score, evaluator, epoch, log_per_step=1):
    batch = 0
    total_sample = 0
    total_loss = 0
    result = 0
    start = time.time()
    for batch_feed_dict in batch_iter():
        batch += 1
        y_pred = exe.run(program, fetch_list=[score], feed=batch_feed_dict)[0]
        input_dict = {
            "y_true": batch_feed_dict["node_label"],
            "y_pred": y_pred[batch_feed_dict["node_index"]]
        }
        result += evaluator.eval(input_dict)["rocauc"]


        if batch % log_per_step == 0:
            print(batch, result/batch)


        num_samples = len(batch_feed_dict["node_index"])
#         total_loss += batch_loss * num_samples
#         total_acc += batch_acc * num_samples
        total_sample += num_samples
    end = time.time()
    print("Epoch {} Valid: Loss={}, Speed(per batch)={}".format(epoch, total_loss/total_sample,
                                                                (end-start)/batch))
    return total_loss/total_sample, result/batch

    
def valid_epoch(batch_iter, exe, program, loss, score, evaluator, epoch, log_per_step=1):
    batch = 0
    total_sample = 0
    result = 0
    total_loss = 0.0
    for batch_feed_dict in batch_iter():
        batch += 1
        batch_loss, y_pred = exe.run(program, fetch_list=[loss, score], feed=batch_feed_dict)
        input_dict = {
            "y_true": batch_feed_dict["node_label"],
#             "y_pred": y_pred[batch_feed_dict["node_index"]]
            "y_pred": y_pred
        }
#         print(evaluator.eval(input_dict))
        result += evaluator.eval(input_dict)["rocauc"]


#         if batch % log_per_step == 0:
#             print(batch, result/batch)


        num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
#         total_acc += batch_acc * num_samples
        total_sample += num_samples

#     print("Epoch {} Valid: Loss={}, Speed(per batch)={}".format(epoch, total_loss/total_sample, (end-start)/batch))
    return total_loss.item()/total_sample, result/batch

    

def run_epoch(batch_iter, exe, program, prefix, model_loss, model_acc, epoch, log_per_step=100):
    """
    已废弃
    """
    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0
    start = time.time()
    for batch_feed_dict in batch_iter():
        batch += 1
        batch_loss, batch_acc = exe.run(program,
                                        fetch_list=[model_loss, model_acc],
                                        feed=batch_feed_dict)

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %s %s-Acc %s" %
                     (batch, prefix, batch_loss, prefix, batch_acc))

        num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
        total_acc += batch_acc * num_samples
        total_sample += num_samples
    end = time.time()

    log.info("%s Epoch %s Loss %.5lf Acc %.5lf Speed(per batch) %.5lf sec" %
             (prefix, epoch, total_loss / total_sample,
              total_acc / total_sample, (end - start) / batch))
    
