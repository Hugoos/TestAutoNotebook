import openml as oml

def findProblemType(data):
    #0 = classification, 1 = regression, 2 = clustering
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True)
    total = 0
    global text_problemType
    uniqueElementsDict = {}
    answer = -1
    if y.size == 0:
        print("Problem type: clustering problem.")
        print("Clustering problems are currently not supported.")
        return "Clustering"
    for item in y:
        total += 1
        uniqueElementsDict[item] = ""
    
    perUnique = len(uniqueElementsDict) / total
    
    if perUnique > 0.05:
        print("Problem type: supervised regression problem.")
        answer = "Supervised Regression"
    else:
        print("Problem type: supervised classification problem.")
        answer = "Supervised Classification"
    return answer

def checkTask(task, problemType, target):
    if task == -1:
        return
    if bool(task.values()):
        taskValues = next(iter(task.values()))
        taskInfo = oml.tasks.get_task(taskValues['task_id'])
        if taskInfo.task_type == problemType and taskInfo.target_name == target:
            print("Using correct task")
        else:
            print("Task does not match standard target")
    else:
        print("Task is empty so cannot be checked, possibly due to a bug in OpenML")
