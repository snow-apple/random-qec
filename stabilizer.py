import numpy as np
import random
from tabulate import tabulate

def sigmaz():
    return np.array([[1,0], [0, -1]])

def sigmax():
    return np.array([[0,1], [1,0]])

def sigmay():
    return np.array([[0,-1], [1,0]])

#conjugate transpose
def ct(matrix):
    return matrix.conj().T


def next(previous, gate):
    return gate@previous@ct(gate)

#tensor
def tensor(ops):
    answer = ops[0]
    for op in ops[1:]:
        answer = np.kron(answer, op)
    return answer




zero = np.array([[1], [0]])
one = np.array([[0], [1]])

def cnot(control, target, n):
    for i in range(n):
        if i == control:
            sub_g = zero@ct(zero)
            sub_e = one@ct(one)
        elif i == target:
            sub_g = np.eye(2)
            sub_e = sigmax()
        else:
            sub_g = np.eye(2)
            sub_e = np.eye(2)
        g_term = sub_g if i == 0 else np.kron(g_term, sub_g)
        e_term = sub_e if i == 0 else np.kron(e_term,sub_e)
        
    return g_term + e_term
            

#applies cnot 
def apply_cnot(string, control, target):
    if control == target:
        return string
    newstring = ''
    if string[control] == '1':
        if string[target] == '1':
            newstring = string[:target]+'0'+string[target+1:]
        else:
            newstring= string[:target]+'1'+string[target+1:]
    else:
        newstring = string
    return newstring

def apply_x(string, target):
    if string[target] == '1':
        string = string[:target] + '0' + string[target+1:]
    elif string[target] == '0':
        string = string[:target] + '1' + string[target+1:]
    return string
        


def returnarray(list):
    newlist = []
    for l in list:
        if l[0] == "0":
            final = zero
        else:
            final = one
        for x in range(len(l) -1):
            if l[x+1] == "0":
                final = np.kron(final, zero)
            else:
                final = np.kron(final, one)
        newlist.append(final)
    return newlist



#random circuit

def randcircuit(n,d):
    number = d
    finallist =[]
    while number > 0:
        if random.randrange(0, 2, 1) == 0:
            finallist.append(('X', (random.randrange(0,n,1),)))
        else:
            first = random.randrange(0, n, 1)
            second = random.randrange(0, n, 1)
            while first == second:
                second = random.randrange(0, n, 1)
            finallist.append(('CX',(first,second)))
        number -= 1
    return finallist
            
def evolve_stabilizers(stabilizers, gates, logicals):
    stabilizerlist = []
    logicallist = []
    for stabilizer in stabilizers:
        for gate in gates:
            stabilizer = evolve(stabilizer, gate)
        stabilizerlist.append(stabilizer)
    for logical in logicals:
        for gate in gates:
            logical = evolve(logical, gate)
        logicallist.append(logical)
    return (stabilizerlist,logicallist)



def evolve_operators(operators, gates):
    operatorlist = []
    for operator in operators:
        for gate in gates:
            operator = evolve(operator, gate)
        operatorlist.append(operator)
    return operatorlist



def evolve(input, gate):
    if gate[0] == 'CX':
        if input[gate[1][0]] == 'I' and input[gate[1][1]] == 'I':
            #II --> II
            return input
        
        elif input[gate[1][0]] == 'Z' and input[gate[1][1]] == 'I':
            #ZI --> ZI
            return input
            
        elif input[gate[1][0]] == 'I' and input[gate[1][1]] == 'Z':
            #IZ --> ZZ
            input = replace_char(input, gate[1][0], 'Z')
            return input
           
        elif input[gate[1][0]] == 'Z' and input[gate[1][1]] == 'Z':
            #ZZ-->IZ
            input = replace_char(input, gate[1][0], 'I')
            return input
        elif input[gate[1][0]] == 'X' and input[gate[1][1]] == 'I':
            #XI --> XX
            input = replace_char(input, gate[1][1], 'X')
            return input
        elif input[gate[1][0]] == 'I' and input[gate[1][1]] == 'X':
            #IX --> IX
            return input
        elif input[gate[1][0]] == 'X' and input[gate[1][1]] == 'X':
            #XX --> XI
            input = replace_char(input, gate[1][1], 'I')
            return input
        elif input[gate[1][0]] == 'X' and input[gate[1][1]] == 'Z':
            raise NotImplementedError("action of CX on XZ is not implemented")
        elif input[gate[1][0]] == 'Z' and input[gate[1][1]] == 'X':
            raise NotImplementedError("action of CX on ZX is not implemented")
            
    elif gate[0] == 'X':
        if input[gate[1][0]] == 'I':
            return input
        elif input[gate[1][0]] == 'Z':
            return input[:-1] + flip_sign(input[-1])
        elif input[gate[1][0]] == 'X':
            return input
    print("hi")
        

def flip_sign(stab_sign):
    if stab_sign == '+':
        return '-'     
    elif stab_sign == '-':
        return '+'
    else:
        raise ValueError("sign must be + or -")    



def replace_char(stab, i, new_char):
    return stab[:i] + new_char +stab[i+1:]




def construct_error(p,n):
    error = np.eye(2)
    for i in range(n):
        if random.randrange(1,100)/100 < p:
            if i == 0:
                error *= sigmax()
            else:
                error = tensor([error,sigmax()])
        else:
            if i != 0:
                error = tensor([error,np.eye(2)])
    return error


#decoding.ipynb
def construct_list_errors(n, m):
    finallist = []

    for i in range(n):
        if len(finallist) == 0:
            finallist.append('I')
            finallist.append('X')
      
        else:
            for j in range(len(finallist)):
                if number_of_errors(finallist[j]) <m:
                    finallist.append(finallist[j] + 'X')

                finallist[j] = finallist[j] + 'I'
    return finallist

def number_of_errors(string):
    x = 0
    for s in string:
        if s =='X':
            x+=1
    return x
    # errors = []
    # for i in range(2*n):
    #     error = construct_error(0.5, 3)
    #     if len(errors) == 0:
    #         errors.append(error)
    #     else: 
    #         z = 0
    #         for x in errors:
    #             y = x - error == np.zeros(n)
    #             if y == True:
    #                 z+=1
    #         if z == 0:
    #             errors.append(error)
    # return(errors)
             
        
#Decoding

#all possible recovery operations with n qubits
def recovery_operations(n):
    finallist = []
    for i in range(n):
        if len(finallist) == 0:
            finallist.append('I')
            finallist.append('X')
        else:
            for j in range(len(finallist)):
                finallist.append(finallist[j] + 'X')
                finallist[j] = finallist[j] + 'I'
    return finallist

#probabilty of each recovery operation
def recovery_probabilities(recov_op, p):
    recoverydict = {}
    for r in recov_op:
        x = 0
        i = 0
        for j in r:
            if j == 'X':
                x+=1
            elif j == 'I':
                i+=1
        prob = ((1-p)**i)*((p)**(x))
        recoverydict.update({r:round(prob,5)})
    return recoverydict

def encoding(initial, n):
    for i in range(n):
        initial = apply_cnot(initial, 0, i)
    return initial

def decoding(state, n):
    for i in range(n):
        state = apply_cnot(state, 0, n-1-i)
    return state

def apply_error_recovery(initial, apply):
    for i in range(len(apply)):
        if apply[i] == 'X':
            initial = apply_x(initial, i)
    return initial


'''finds all possible recoveries that could work with a given error'''
def recovery(initial, error, p):
    n = len(initial)
    encodingstate = encoding(initial,n)
    errorstate = apply_error_recovery(encodingstate, error)
    rlist = recovery_probabilities(recovery_operations(n), p)
    successfulrecoveries = {}
    for r in rlist:
        recoverystate = apply_error_recovery(errorstate, r)
        decodingstate = decoding(recoverystate,n)
        #checking if every qubit except first is 0
        count = 0
        for i in range(n-1): 
            if decodingstate[n-1-i] != '0':
                count+=1
        if count == 0:
            successfulrecoveries.update({r:rlist[r]}) 

    return successfulrecoveries

#returns a table of errors with corrresponding recoveries
def table_of_errors(initial):
    n = len(initial)
    elist = construct_list_errors(n,1)
    table = {}
    for e in elist:
        rlist = recovery(initial, e)
        highestprob = 0
        highestr = ''
        for r in rlist:
            if rlist[r] > highestprob:
                highestprob = rlist[r]
                highestr = r
        table.update({e:highestr})
    return table

def print_table(table):
    result = table.items()
    data = list(result)
    numpyArray = np.array(data)
    head = ['Error', 'Recovery']
    print(tabulate(numpyArray, headers=head, tablefmt="grid"))



#random code simulation
def find_codewords(initial,gates):
    result=[]
    for state in initial:
        for x in gates:
            if x[0] == 'X':
                state = apply_x(state,x[1][0])
            elif x[0] == 'CX':
                state = apply_cnot(state,x[1][0], x[1][1])
        result.append(state)
    return result 


def measure_stabilizers(stabilizers, codeword):
    results = []
    for stab in stabilizers:
        result = 1
        for i in range(len(stab)-1):
            if stab[i] == 'Z' and codeword[i] == '1':
                result *= -1 
        results.append(result)
        
    return results
            

    
def single_recovery_probability(recov_op, p):
    recoverydict = {}
    x = 0
    i = 0
    for r in recov_op:
        if r == 'X':
            x+=1
        elif r == 'I':
            i+=1
    prob = ((1-p)**i)*((p)**(x))

    return prob