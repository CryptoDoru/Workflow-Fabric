import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Agents from './pages/Agents'
import Workflows from './pages/Workflows'
import WorkflowBuilder from './pages/WorkflowBuilder'
import Executions from './pages/Executions'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="agents" element={<Agents />} />
        <Route path="workflows" element={<Workflows />} />
        <Route path="workflows/builder" element={<WorkflowBuilder />} />
        <Route path="workflows/builder/:id" element={<WorkflowBuilder />} />
        <Route path="executions" element={<Executions />} />
      </Route>
    </Routes>
  )
}

export default App
