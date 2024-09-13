graph = {
    '5':['3','7'],
    '3':['2','4'],
    '7':['8'],
    '2':[],
    '4':['8'],
    '8':[]
}
visited = [] 
queue = [] 
def bfs(visited, graph, root,des):
    visited.append(root)
    queue.append(root)
    while queue:
        m=queue.pop(0)
        print(m,end=(" "))
        if m == des :
            break
        for n in graph[m]:
            if n  not in visited:
                visited.append(n)
                queue.append(n)


print("Following is the Breadth-First Search")
bfs(visited, graph, '5', '7') 
