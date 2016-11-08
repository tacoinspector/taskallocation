import pandas as pd
from pulp import *
import itertools as it


FILENAME = "~/Documents/taskallocation.xlsx"

def getPersonToSkill(fileName):
    return pd.read_excel(fileName, "PersonToSkill").fillna(0).astype('int')

def getDeadlines(fileName):
    return pd.read_excel(fileName, "Deadline").fillna(1e3).set_index('Project')
    

def getProjectToSkillRequirements(fileName):
    return pd.read_excel(fileName, "ProjectSkillRequirements").fillna(.0).astype('int')

def getImmediatePrecedents(fileName):
    df = pd.read_excel(fileName, "ImmediatePrecedent")
    prec = df['Project'] + '_' + df['Skill']
    dep = df['DepProject'] + '_' + df['DepSkill']
    return pd.DataFrame({'Dep': dep, 'Prec': prec}).set_index('Prec')
    

def getStartTimes(fileName):
    return pd.read_excel(fileName, "StartTimes").fillna(.0).astype('int')
    
    
def getPersonToSkillMatrix(personToSkill, tasks):
    def getPersonToSkillVector(skills, taskTuple):
        validSkills = list(skills[skills>0].index)
        return [1 if y in validSkills else 0 for x, y in taskTuple]
    taskTuple = [y.split('_') for y in tasks]
    ret = {}
    for person in personToSkill.index:
        ret[person] = dict(zip(tasks, getPersonToSkillVector(personToSkill.loc[person], taskTuple)))
    return ret
    
    
def getPersonTimeToCompleteMatrix(personToSkill, projectToSkill, tasks):
    def getPersonToTimeVector(skills, taskTuple):
        validSkills = list(skills[skills>0].index)
        return np.array([int(projectToSkill.loc[x, y]) if y in validSkills else 0 for x, y in taskTuple])
    taskTuple = [y.split('_') for y in tasks]
        
    ret = {}
    for person in personToSkill.index:
        ret[person] = dict(zip(tasks, getPersonToTimeVector(personToSkill.loc[person], taskTuple)))
    return ret
    
def getTaskTimeExpr(task, people, personTimeMatrix, taskAllocationMatrix):
    return lpSum(taskAllocationMatrix[(task, person)] * personTimeMatrix[person][task] for person in people)
    

personToSkill = getPersonToSkill(FILENAME)
projectToSkill = getProjectToSkillRequirements(FILENAME)
deadlines = getDeadlines(FILENAME)
immedPrec = getImmediatePrecedents(FILENAME)

people = list(personToSkill.index)
projects = list(projectToSkill.index)
skills = list(projectToSkill.columns)
taskTuple = list(it.product(projects, skills))
tasks = ['{}_{}'.format(x, y) for x, y in taskTuple]
taskToTuple = dict(zip(tasks, taskTuple))
reqdTimePerTask = [projectToSkill.loc[x, y] for x, y in taskTuple]
tasks = [t for t, rtt in zip(tasks, reqdTimePerTask) if rtt > 0]



taskAllocationMatrix = LpVariable.dicts("TaskAllocationMatrix", [(i, j) for i in tasks for j in people],cat=LpBinary)
overlapTimeSpent = LpVariable.dicts("OverlapIndicator", [(i, j) for i in tasks for j in tasks], cat=LpBinary)
startTimePerTask = LpVariable.dicts('StartTimePerTask', tasks, 0, int(projectToSkill.sum(axis=1).sum()), cat=LpInteger)
maxTime = LpVariable("maxTime", lowBound =0, upBound = 10000, cat=LpInteger)
timePerTask = LpVariable.dict('timePerTask',  tasks, lowBound = 0, upBound = 10000, cat = LpInteger)


personAbilityMatrix = getPersonToSkillMatrix(personToSkill, tasks)
personTimeMatrix = getPersonTimeToCompleteMatrix(personToSkill, projectToSkill, tasks)

taskAllocation = pulp.LpProblem("Task allocation", LpMinimize)


# Minimise the average time to complete tasks - can add weighting
#taskAllocation += lpSum(timePerTask)
taskAllocation += maxTime

for task in tasks:
    tt = taskToTuple[task]
    # Set the timePerTask to the start time + the time to complete
    taskAllocation += timePerTask[task] == getTaskTimeExpr(task, people, personTimeMatrix, taskAllocationMatrix) + startTimePerTask[task]
    taskAllocation += maxTime >= timePerTask[task]
    # Only allow people to work on tasks they can work on    
    for person in people:
        if personAbilityMatrix[person][task] < 1:
            taskAllocation += taskAllocationMatrix[(task, person)] == 0
    
    # Make sure all tasks with work are allocated to exactly one person
    if projectToSkill.loc[tt[0], tt[1]] > 0:
        taskAllocation += lpSum(taskAllocationMatrix[(task, person)] for person in people) == 1
        
    # Work deadlines
    #taskAllocation += getTaskTimeExpr(task, people, personTimeMatrix, taskAllocationMatrix) + startTimePerTask[task] <= deadlines.loc[tt[0], 'Deadline']

# Ensure no overlap
if True:
    for task_k in tasks:
        for task_j in tasks:
            if task_k != task_j: 
                taskAllocation += startTimePerTask[task_k] - getTaskTimeExpr(task_j, people, personTimeMatrix, taskAllocationMatrix) - startTimePerTask[task_j] >= -100000 * overlapTimeSpent[(task_j, task_k)]
                    
                taskAllocation += -startTimePerTask[task_k] + \
                    getTaskTimeExpr(task_j, people, personTimeMatrix, taskAllocationMatrix) + \
                    startTimePerTask[task_j] >= 100000 * (overlapTimeSpent[(task_j, task_k)] - 1)
                
            for person in people:
                if personAbilityMatrix[person][task_k] > 0 and personAbilityMatrix[person][task_j] > 0:
                    taskAllocation += taskAllocationMatrix[(task_j, person)] + taskAllocationMatrix[(task_k, person)] \
                        + overlapTimeSpent[(task_j, task_k)] + overlapTimeSpent[(task_k, task_j)] <= 3

if True:
    # Take care of dependencies
    # j is prec, k is dep
    for prec, dep in immedPrec['Dep'].iteritems():
        precTt = taskToTuple[prec]
        depTt = taskToTuple[dep]
        for person_a in people:
            if personToSkill.loc[person_a, precTt[1]] < 1: 
                continue
            for person_b in people:
                if personToSkill.loc[person_b, depTt[1]] < 1:
                    continue
            
                taskAllocation += startTimePerTask[dep] >= startTimePerTask[prec]
                taskAllocation += startTimePerTask[dep] >= startTimePerTask[prec] + \
                    (personTimeMatrix[person_a][prec]) * (taskAllocationMatrix[(prec, person_a)] + taskAllocationMatrix[(dep, person_b)] - 1)


taskAllocation.solve()
if taskAllocation.status == 1:
    endTime = 0
    for task in tasks:
        for person in people:
            v = taskAllocationMatrix[(task, person)].varValue
            if np.abs(v - 1) < .005:
                tt = taskToTuple[task]
                timeToEnd =  v * projectToSkill.loc[tt[0], tt[1]] + startTimePerTask[task].varValue
                timeToEnd = max(endTime, timeToEnd)
                print '{} {} {} {}'.format(task, person, v * projectToSkill.loc[tt[0], tt[1]], startTimePerTask[task].varValue)
    print 'Time to finish: {}'.format(timeToEnd)