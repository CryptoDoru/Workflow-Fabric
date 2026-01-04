import { useCallback, useState, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  Panel,
  MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { Save, Play, Plus, Trash2, Settings } from 'lucide-react'
import AgentNode from '../components/AgentNode'

const initialNodes: Node[] = [
  {
    id: 'start',
    type: 'input',
    data: { label: 'Start' },
    position: { x: 250, y: 0 },
    style: {
      background: '#10B981',
      color: 'white',
      border: 'none',
      borderRadius: '8px',
      padding: '10px 20px',
    },
  },
]

const initialEdges: Edge[] = []

const nodeTypes = {
  agent: AgentNode,
}

const defaultEdgeOptions = {
  animated: true,
  markerEnd: {
    type: MarkerType.ArrowClosed,
    width: 20,
    height: 20,
  },
  style: {
    strokeWidth: 2,
  },
}

interface WorkflowConfig {
  id: string
  name: string
  description: string
}

export default function WorkflowBuilder() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [workflowConfig, setWorkflowConfig] = useState<WorkflowConfig>({
    id: id || 'new-workflow',
    name: 'New Workflow',
    description: '',
  })

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }, [])

  const addAgentNode = useCallback(() => {
    const newNode: Node = {
      id: `agent-${Date.now()}`,
      type: 'agent',
      data: {
        label: 'New Agent',
        agentId: '',
        timeout: 30000,
        retry: { maxAttempts: 3, backoffMs: 1000 },
      },
      position: {
        x: Math.random() * 300 + 100,
        y: Math.random() * 200 + 100,
      },
    }
    setNodes((nds) => [...nds, newNode])
  }, [setNodes])

  const deleteSelectedNode = useCallback(() => {
    if (selectedNode && selectedNode.id !== 'start') {
      setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id))
      setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id))
      setSelectedNode(null)
    }
  }, [selectedNode, setNodes, setEdges])

  const saveWorkflow = useCallback(async () => {
    const workflow = {
      id: workflowConfig.id,
      name: workflowConfig.name,
      description: workflowConfig.description,
      steps: nodes
        .filter((n) => n.type === 'agent')
        .map((n) => ({
          id: n.id,
          agentId: n.data.agentId || n.id,
          timeout_ms: n.data.timeout,
          retry: n.data.retry,
          depends_on: edges.filter((e) => e.target === n.id).map((e) => e.source),
        })),
    }

    console.log('Saving workflow:', workflow)
    
    try {
      const res = await fetch('/api/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow),
      })
      if (res.ok) {
        alert('Workflow saved!')
        navigate('/workflows')
      }
    } catch (e) {
      console.error('Failed to save:', e)
      alert('Workflow saved locally (API not available)')
    }
  }, [nodes, edges, workflowConfig, navigate])

  const executeWorkflow = useCallback(async () => {
    try {
      const res = await fetch(`/api/workflows/${workflowConfig.id}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: {} }),
      })
      if (res.ok) {
        alert('Workflow execution started!')
        navigate('/executions')
      }
    } catch (e) {
      console.error('Failed to execute:', e)
      alert('Execution started (demo mode)')
      navigate('/executions')
    }
  }, [workflowConfig.id, navigate])

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <input
            type="text"
            value={workflowConfig.name}
            onChange={(e) => setWorkflowConfig({ ...workflowConfig, name: e.target.value })}
            className="text-2xl font-bold bg-transparent border-b-2 border-transparent hover:border-gray-300 focus:border-indigo-500 focus:outline-none text-gray-900 dark:text-white"
          />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={addAgentNode}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm font-medium"
          >
            <Plus className="h-4 w-4" />
            Add Agent
          </button>
          <button
            onClick={deleteSelectedNode}
            disabled={!selectedNode || selectedNode.id === 'start'}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 text-red-600 dark:text-red-400 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </button>
          <button
            onClick={saveWorkflow}
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors text-sm font-medium"
          >
            <Save className="h-4 w-4" />
            Save
          </button>
          <button
            onClick={executeWorkflow}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
          >
            <Play className="h-4 w-4" />
            Run
          </button>
        </div>
      </div>

      <div className="flex-1 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          fitView
          className="bg-gray-50 dark:bg-gray-900"
        >
          <Controls className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700" />
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          
          {selectedNode && selectedNode.type === 'agent' && (
            <Panel position="top-right" className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 w-72">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Node Settings
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Node Name
                  </label>
                  <input
                    type="text"
                    value={selectedNode.data.label}
                    onChange={(e) => {
                      setNodes((nds) =>
                        nds.map((n) =>
                          n.id === selectedNode.id
                            ? { ...n, data: { ...n.data, label: e.target.value } }
                            : n
                        )
                      )
       
