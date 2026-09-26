/*
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║  PS2 — Federated Learning for Cross-Institution Financial Risk Control     ║
 * ║  Java Swing Dashboard: Real-Time Federated Network Visualization           ║
 * ║  Team JabsonX: Saksham, Stanley, Tanushri, Alwyn, Augustine               ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 *
 * This dashboard provides real-time visualization of:
 *   1. Three institutional nodes (Bank, Lending App, Insurer)
 *   2. Network traffic between nodes and the aggregator
 *   3. Final combined risk scores for queried customers
 *   4. Visual alerts when a malicious node update is intercepted
 *
 * Compile & Run:
 *   javac FederatedDashboard.java
 *   java FederatedDashboard
 *
 * NOTE: This reads results from the Python backend's output files.
 *       Run the Python pipeline first to generate data.
 */

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.table.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.Timer;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Main dashboard application for the Federated Ensemble Scoring network.
 */
public class FederatedDashboard extends JFrame {

    // ═══════════════════════════════════════════════════════════════════
    // THEME CONSTANTS
    // ═══════════════════════════════════════════════════════════════════
    
    // Dark theme palette
    private static final Color BG_PRIMARY     = new Color(13, 17, 23);
    private static final Color BG_SECONDARY   = new Color(22, 27, 34);
    private static final Color BG_TERTIARY    = new Color(33, 38, 45);
    private static final Color BORDER_COLOR   = new Color(48, 54, 61);
    private static final Color TEXT_PRIMARY    = new Color(230, 237, 243);
    private static final Color TEXT_SECONDARY  = new Color(139, 148, 158);
    private static final Color TEXT_MUTED      = new Color(110, 118, 129);
    
    // Accent colors for institutions
    private static final Color BANK_COLOR       = new Color(56, 139, 253);   // Blue
    private static final Color LENDING_COLOR    = new Color(163, 113, 247);  // Purple
    private static final Color INSURER_COLOR    = new Color(63, 185, 80);    // Green
    private static final Color AGGREGATOR_COLOR = new Color(255, 166, 87);   // Orange
    
    // Alert colors
    private static final Color ALERT_INFO     = new Color(56, 139, 253);
    private static final Color ALERT_WARNING  = new Color(210, 153, 34);
    private static final Color ALERT_CRITICAL = new Color(248, 81, 73);
    private static final Color ALERT_BLOCKED  = new Color(218, 54, 51);
    
    // Risk tier colors
    private static final Color RISK_LOW      = new Color(63, 185, 80);
    private static final Color RISK_MEDIUM   = new Color(210, 153, 34);
    private static final Color RISK_HIGH     = new Color(248, 81, 73);
    private static final Color RISK_CRITICAL = new Color(218, 54, 51);
    
    // Fonts
    private static final Font FONT_TITLE   = new Font("Segoe UI", Font.BOLD, 18);
    private static final Font FONT_HEADING = new Font("Segoe UI", Font.BOLD, 14);
    private static final Font FONT_BODY    = new Font("Segoe UI", Font.PLAIN, 12);
    private static final Font FONT_MONO    = new Font("Consolas", Font.PLAIN, 11);
    private static final Font FONT_SMALL   = new Font("Segoe UI", Font.PLAIN, 10);
    
    // ═══════════════════════════════════════════════════════════════════
    // DATA MODEL
    // ═══════════════════════════════════════════════════════════════════
    
    /** Represents a node in the federated network */
    static class NodeInfo {
        String name;
        String shortName;
        String type;
        Color color;
        double trustScore = 1.0;
        boolean isBlocked = false;
        int scoresSent = 0;
        int scoresRejected = 0;
        String status = "ACTIVE";
        List<String> recentLogs = new CopyOnWriteArrayList<>();
        
        NodeInfo(String name, String shortName, String type, Color color) {
            this.name = name;
            this.shortName = shortName;
            this.type = type;
            this.color = color;
        }
    }
    
    /** Represents a network traffic event */
    static class TrafficEvent {
        long timestamp;
        String source;
        String destination;
        String eventType;   // "SCORE_REQUEST", "SCORE_RESPONSE", "ALERT", "BLOCKED"
        String message;
        Color color;
        
        TrafficEvent(String source, String dest, String type, String msg, Color color) {
            this.timestamp = System.currentTimeMillis();
            this.source = source;
            this.destination = dest;
            this.eventType = type;
            this.message = msg;
            this.color = color;
        }
    }
    
    /** Customer risk result */
    static class RiskResult {
        String hashedId;
        double finalScore;
        String riskTier;
        Map<String, Double> nodeScores = new LinkedHashMap<>();
        int alertCount = 0;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════
    
    private Map<String, NodeInfo> nodes = new LinkedHashMap<>();
    private List<TrafficEvent> trafficLog = new CopyOnWriteArrayList<>();
    private List<RiskResult> riskResults = new CopyOnWriteArrayList<>();
    private List<String> alertLog = new CopyOnWriteArrayList<>();
    
    // UI Components
    private NetworkPanel networkPanel;
    private JTextArea trafficTextArea;
    private JTable riskTable;
    private DefaultTableModel riskTableModel;
    private JTextArea alertTextArea;
    private JLabel statusLabel;
    private JProgressBar progressBar;
    private JPanel alertFlashPanel;
    private Timer simulationTimer;
    
    // Simulation state
    private boolean isSimulating = false;
    private int simulationStep = 0;
    private static final int TOTAL_SIMULATION_STEPS = 50;
    
    // ═══════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════
    
    public FederatedDashboard() {
        super("PS2 — Federated Ensemble Scoring Dashboard | Team JabsonX");
        
        initializeNodes();
        initializeUI();
        
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1400, 900);
        setMinimumSize(new Dimension(1100, 750));
        setLocationRelativeTo(null);
        
        // Set dark theme for entire application
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            // Fallback to default
        }
        
        setVisible(true);
    }
    
    private void initializeNodes() {
        nodes.put("bank", new NodeInfo(
            "Bank Node (RBI-Regulated)", "BANK", "bank", BANK_COLOR
        ));
        nodes.put("lending_app", new NodeInfo(
            "Lending App (Alt-Data)", "LEND", "lending_app", LENDING_COLOR
        ));
        nodes.put("insurer", new NodeInfo(
            "Insurer Node (IRDAI)", "INSR", "insurer", INSURER_COLOR
        ));
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // UI INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════
    
    private void initializeUI() {
        JPanel mainPanel = new JPanel(new BorderLayout(0, 0));
        mainPanel.setBackground(BG_PRIMARY);
        
        // ── Top Bar ──
        mainPanel.add(createTopBar(), BorderLayout.NORTH);
        
        // ── Main Content (split pane) ──
        JSplitPane mainSplit = new JSplitPane(
            JSplitPane.HORIZONTAL_SPLIT,
            createLeftPanel(),
            createRightPanel()
        );
        mainSplit.setDividerLocation(750);
        mainSplit.setDividerSize(3);
        mainSplit.setBorder(null);
        mainSplit.setBackground(BG_PRIMARY);
        
        mainPanel.add(mainSplit, BorderLayout.CENTER);
        
        // ── Bottom Status Bar ──
        mainPanel.add(createStatusBar(), BorderLayout.SOUTH);
        
        setContentPane(mainPanel);
    }
    
    /** Creates the top header bar with title and controls */
    private JPanel createTopBar() {
        JPanel topBar = new JPanel(new BorderLayout());
        topBar.setBackground(BG_SECONDARY);
        topBar.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(0, 0, 1, 0, BORDER_COLOR),
            BorderFactory.createEmptyBorder(10, 15, 10, 15)
        ));
        
        // Title
        JLabel titleLabel = new JLabel("⬡ Federated Ensemble Scoring — Live Dashboard");
        titleLabel.setFont(FONT_TITLE);
        titleLabel.setForeground(TEXT_PRIMARY);
        topBar.add(titleLabel, BorderLayout.WEST);
        
        // Control buttons
        JPanel controls = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        controls.setOpaque(false);
        
        JButton simulateBtn = createStyledButton("▶ Run Simulation", AGGREGATOR_COLOR);
        simulateBtn.addActionListener(e -> startSimulation());
        
        JButton attackBtn = createStyledButton("⚡ Simulate Attack", ALERT_CRITICAL);
        attackBtn.addActionListener(e -> simulateAttack());
        
        JButton resetBtn = createStyledButton("↻ Reset", TEXT_SECONDARY);
        resetBtn.addActionListener(e -> resetDashboard());
        
        JButton loadBtn = createStyledButton("📂 Load Results", ALERT_INFO);
        loadBtn.addActionListener(e -> loadResultsFromFile());
        
        controls.add(loadBtn);
        controls.add(simulateBtn);
        controls.add(attackBtn);
        controls.add(resetBtn);
        
        topBar.add(controls, BorderLayout.EAST);
        
        // Alert flash panel (invisible until alert)
        alertFlashPanel = new JPanel();
        alertFlashPanel.setPreferredSize(new Dimension(0, 3));
        alertFlashPanel.setBackground(BG_SECONDARY);
        topBar.add(alertFlashPanel, BorderLayout.SOUTH);
        
        return topBar;
    }
    
    /** Creates the left panel (network viz + traffic log) */
    private JPanel createLeftPanel() {
        JPanel left = new JPanel(new BorderLayout(0, 0));
        left.setBackground(BG_PRIMARY);
        
        // Network visualization (top)
        networkPanel = new NetworkPanel();
        networkPanel.setPreferredSize(new Dimension(700, 400));
        networkPanel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(0, 0, 1, 1, BORDER_COLOR),
            BorderFactory.createEmptyBorder(5, 5, 5, 5)
        ));
        
        // Traffic log (bottom)
        JPanel trafficPanel = createTrafficLogPanel();
        
        JSplitPane leftSplit = new JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            networkPanel,
            trafficPanel
        );
        leftSplit.setDividerLocation(400);
        leftSplit.setDividerSize(3);
        leftSplit.setBorder(null);
        
        left.add(leftSplit, BorderLayout.CENTER);
        return left;
    }
    
    /** Creates the right panel (risk scores + alerts) */
    private JPanel createRightPanel() {
        JPanel right = new JPanel(new BorderLayout(0, 0));
        right.setBackground(BG_PRIMARY);
        
        // Node status cards (top)
        JPanel nodeCards = createNodeStatusCards();
        
        // Risk scores table (middle)
        JPanel riskPanel = createRiskScorePanel();
        
        // Alerts panel (bottom)
        JPanel alertPanel = createAlertPanel();
        
        JSplitPane rightSplit = new JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            riskPanel,
            alertPanel
        );
        rightSplit.setDividerLocation(350);
        rightSplit.setDividerSize(3);
        rightSplit.setBorder(null);
        
        JPanel rightInner = new JPanel(new BorderLayout(0, 0));
        rightInner.setBackground(BG_PRIMARY);
        rightInner.add(nodeCards, BorderLayout.NORTH);
        rightInner.add(rightSplit, BorderLayout.CENTER);
        
        right.add(rightInner, BorderLayout.CENTER);
        return right;
    }
    
    /** Creates node status cards showing trust scores and status */
    private JPanel createNodeStatusCards() {
        JPanel cardsPanel = new JPanel(new GridLayout(1, 3, 6, 0));
        cardsPanel.setBackground(BG_PRIMARY);
        cardsPanel.setBorder(BorderFactory.createEmptyBorder(8, 8, 4, 8));
        
        for (NodeInfo node : nodes.values()) {
            cardsPanel.add(createNodeCard(node));
        }
        
        return cardsPanel;
    }
    
    /** Creates a single node status card */
    private JPanel createNodeCard(NodeInfo node) {
        JPanel card = new JPanel(new BorderLayout(4, 4));
        card.setBackground(BG_TERTIARY);
        card.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(node.color.darker(), 1),
            BorderFactory.createEmptyBorder(8, 10, 8, 10)
        ));
        
        // Header
        JLabel nameLabel = new JLabel("● " + node.shortName);
        nameLabel.setFont(FONT_HEADING);
        nameLabel.setForeground(node.color);
        card.add(nameLabel, BorderLayout.NORTH);
        
        // Stats
        JPanel statsPanel = new JPanel(new GridLayout(3, 1, 0, 2));
        statsPanel.setOpaque(false);
        
        JLabel trustLabel = new JLabel(String.format("Trust: %.0f%%", node.trustScore * 100));
        trustLabel.setFont(FONT_SMALL);
        trustLabel.setForeground(node.trustScore > 0.5 ? RISK_LOW : RISK_HIGH);
        
        JLabel scoresLabel = new JLabel("Scores: " + node.scoresSent);
        scoresLabel.setFont(FONT_SMALL);
        scoresLabel.setForeground(TEXT_SECONDARY);
        
        JLabel statusNodeLabel = new JLabel("Status: " + node.status);
        statusNodeLabel.setFont(FONT_SMALL);
        statusNodeLabel.setForeground(node.isBlocked ? RISK_CRITICAL : RISK_LOW);
        
        statsPanel.add(trustLabel);
        statsPanel.add(scoresLabel);
        statsPanel.add(statusNodeLabel);
        
        card.add(statsPanel, BorderLayout.CENTER);
        
        return card;
    }
    
    /** Creates the traffic log panel */
    private JPanel createTrafficLogPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(BG_SECONDARY);
        panel.setBorder(BorderFactory.createMatteBorder(0, 0, 0, 1, BORDER_COLOR));
        
        JLabel header = new JLabel("  📡 Network Traffic Log");
        header.setFont(FONT_HEADING);
        header.setForeground(TEXT_PRIMARY);
        header.setBorder(BorderFactory.createEmptyBorder(6, 4, 6, 4));
        header.setOpaque(true);
        header.setBackground(BG_TERTIARY);
        
        trafficTextArea = new JTextArea();
        trafficTextArea.setBackground(BG_PRIMARY);
        trafficTextArea.setForeground(TEXT_SECONDARY);
        trafficTextArea.setFont(FONT_MONO);
        trafficTextArea.setEditable(false);
        trafficTextArea.setLineWrap(true);
        trafficTextArea.setWrapStyleWord(true);
        trafficTextArea.setCaretColor(TEXT_PRIMARY);
        
        JScrollPane scrollPane = new JScrollPane(trafficTextArea);
        scrollPane.setBorder(null);
        scrollPane.getVerticalScrollBar().setUnitIncrement(12);
        
        panel.add(header, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    /** Creates the risk score table panel */
    private JPanel createRiskScorePanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(BG_SECONDARY);
        
        JLabel header = new JLabel("  📊 Aggregated Risk Scores");
        header.setFont(FONT_HEADING);
        header.setForeground(TEXT_PRIMARY);
        header.setBorder(BorderFactory.createEmptyBorder(6, 4, 6, 4));
        header.setOpaque(true);
        header.setBackground(BG_TERTIARY);
        
        String[] columns = {
            "Customer ID (Hash)", "Final Score", "Risk Tier",
            "Bank", "Lending", "Insurer", "Confidence"
        };
        riskTableModel = new DefaultTableModel(columns, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };
        
        riskTable = new JTable(riskTableModel);
        riskTable.setBackground(BG_PRIMARY);
        riskTable.setForeground(TEXT_PRIMARY);
        riskTable.setFont(FONT_MONO);
        riskTable.setGridColor(BORDER_COLOR);
        riskTable.setSelectionBackground(BG_TERTIARY);
        riskTable.setSelectionForeground(TEXT_PRIMARY);
        riskTable.setRowHeight(24);
        riskTable.setShowGrid(true);
        riskTable.setIntercellSpacing(new Dimension(1, 1));
        
        // Custom header
        JTableHeader tableHeader = riskTable.getTableHeader();
        tableHeader.setBackground(BG_TERTIARY);
        tableHeader.setForeground(TEXT_PRIMARY);
        tableHeader.setFont(FONT_BODY);
        tableHeader.setBorder(BorderFactory.createMatteBorder(0, 0, 1, 0, BORDER_COLOR));
        
        // Custom cell renderer for risk tier coloring
        riskTable.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(
                JTable table, Object value, boolean isSelected,
                boolean hasFocus, int row, int column
            ) {
                Component c = super.getTableCellRendererComponent(
                    table, value, isSelected, hasFocus, row, column
                );
                
                c.setBackground(isSelected ? BG_TERTIARY : BG_PRIMARY);
                c.setForeground(TEXT_PRIMARY);
                
                if (column == 2 && value != null) {  // Risk Tier column
                    String tier = value.toString();
                    switch (tier) {
                        case "LOW":      c.setForeground(RISK_LOW);      break;
                        case "MEDIUM":   c.setForeground(RISK_MEDIUM);   break;
                        case "HIGH":     c.setForeground(RISK_HIGH);     break;
                        case "CRITICAL": c.setForeground(RISK_CRITICAL); break;
                    }
                }
                
                if (column == 1 && value != null) {  // Final Score column
                    try {
                        double score = Double.parseDouble(value.toString());
                        if (score >= 0.75) c.setForeground(RISK_CRITICAL);
                        else if (score >= 0.50) c.setForeground(RISK_HIGH);
                        else if (score >= 0.25) c.setForeground(RISK_MEDIUM);
                        else c.setForeground(RISK_LOW);
                    } catch (NumberFormatException e) { /* ignore */ }
                }
                
                return c;
            }
        });
        
        JScrollPane scrollPane = new JScrollPane(riskTable);
        scrollPane.setBorder(null);
        scrollPane.getViewport().setBackground(BG_PRIMARY);
        
        panel.add(header, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    /** Creates the alerts panel */
    private JPanel createAlertPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(BG_SECONDARY);
        
        JLabel header = new JLabel("  🛡 Security Alerts & Poisoning Defense");
        header.setFont(FONT_HEADING);
        header.setForeground(ALERT_CRITICAL);
        header.setBorder(BorderFactory.createEmptyBorder(6, 4, 6, 4));
        header.setOpaque(true);
        header.setBackground(BG_TERTIARY);
        
        alertTextArea = new JTextArea();
        alertTextArea.setBackground(new Color(30, 15, 15));
        alertTextArea.setForeground(ALERT_CRITICAL);
        alertTextArea.setFont(FONT_MONO);
        alertTextArea.setEditable(false);
        alertTextArea.setLineWrap(true);
        alertTextArea.setWrapStyleWord(true);
        
        JScrollPane scrollPane = new JScrollPane(alertTextArea);
        scrollPane.setBorder(null);
        
        panel.add(header, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    /** Creates the bottom status bar */
    private JPanel createStatusBar() {
        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBackground(BG_TERTIARY);
        statusBar.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(1, 0, 0, 0, BORDER_COLOR),
            BorderFactory.createEmptyBorder(4, 10, 4, 10)
        ));
        
        statusLabel = new JLabel("Ready — Click 'Run Simulation' or 'Load Results' to begin");
        statusLabel.setFont(FONT_SMALL);
        statusLabel.setForeground(TEXT_SECONDARY);
        
        progressBar = new JProgressBar(0, 100);
        progressBar.setPreferredSize(new Dimension(200, 16));
        progressBar.setStringPainted(true);
        progressBar.setFont(FONT_SMALL);
        progressBar.setForeground(AGGREGATOR_COLOR);
        progressBar.setBackground(BG_PRIMARY);
        progressBar.setValue(0);
        progressBar.setVisible(false);
        
        JLabel teamLabel = new JLabel("Team JabsonX © 2026  ");
        teamLabel.setFont(FONT_SMALL);
        teamLabel.setForeground(TEXT_MUTED);
        
        statusBar.add(statusLabel, BorderLayout.WEST);
        statusBar.add(progressBar, BorderLayout.CENTER);
        statusBar.add(teamLabel, BorderLayout.EAST);
        
        return statusBar;
    }
    
    /** Creates a styled button */
    private JButton createStyledButton(String text, Color accentColor) {
        JButton button = new JButton(text);
        button.setFont(FONT_BODY);
        button.setForeground(TEXT_PRIMARY);
        button.setBackground(BG_TERTIARY);
        button.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(accentColor.darker(), 1),
            BorderFactory.createEmptyBorder(5, 12, 5, 12)
        ));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        
        button.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent e) {
                button.setBackground(accentColor.darker().darker());
            }
            @Override
            public void mouseExited(MouseEvent e) {
                button.setBackground(BG_TERTIARY);
            }
        });
        
        return button;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // NETWORK VISUALIZATION PANEL
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * Custom panel that renders the federated network topology:
     * - Three institution nodes arranged in a triangle
     * - Central aggregator node
     * - Animated data flow lines between nodes
     * - Visual indicators for trust scores and block status
     */
    class NetworkPanel extends JPanel {
        
        private List<AnimatedParticle> particles = new CopyOnWriteArrayList<>();
        private Timer animationTimer;
        private int frameCount = 0;
        
        NetworkPanel() {
            setBackground(BG_PRIMARY);
            
            // Animation timer (30 FPS)
            animationTimer = new Timer();
            animationTimer.scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    frameCount++;
                    updateParticles();
                    repaint();
                }
            }, 0, 33);
        }
        
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_LCD_HRGB);
            
            int w = getWidth();
            int h = getHeight();
            
            // Background gradient
            GradientPaint bgGrad = new GradientPaint(0, 0, BG_PRIMARY, w, h, BG_SECONDARY);
            g2.setPaint(bgGrad);
            g2.fillRect(0, 0, w, h);
            
            // Grid pattern
            g2.setColor(new Color(40, 44, 50, 30));
            for (int x = 0; x < w; x += 30) g2.drawLine(x, 0, x, h);
            for (int y = 0; y < h; y += 30) g2.drawLine(0, y, w, y);
            
            // Node positions
            int cx = w / 2;
            int cy = h / 2;
            int radius = Math.min(w, h) / 3;
            
            int[][] nodePositions = {
                {cx - radius, cy - radius / 2},       // Bank (top-left)
                {cx + radius, cy - radius / 2},       // Lending (top-right)
                {cx, cy + radius},                     // Insurer (bottom)
            };
            
            // Draw connections
            NodeInfo[] nodeArray = nodes.values().toArray(new NodeInfo[0]);
            for (int i = 0; i < nodePositions.length; i++) {
                drawConnection(g2, nodePositions[i][0], nodePositions[i][1],
                             cx, cy, nodeArray[i].color, nodeArray[i].isBlocked);
            }
            
            // Draw particles
            for (AnimatedParticle p : particles) {
                g2.setColor(new Color(p.color.getRed(), p.color.getGreen(),
                           p.color.getBlue(), (int)(p.alpha * 255)));
                g2.fillOval((int)p.x - 4, (int)p.y - 4, 8, 8);
                
                // Glow
                g2.setColor(new Color(p.color.getRed(), p.color.getGreen(),
                           p.color.getBlue(), (int)(p.alpha * 80)));
                g2.fillOval((int)p.x - 8, (int)p.y - 8, 16, 16);
            }
            
            // Draw aggregator (center)
            drawAggregatorNode(g2, cx, cy);
            
            // Draw institution nodes
            for (int i = 0; i < nodePositions.length; i++) {
                drawInstitutionNode(g2, nodePositions[i][0], nodePositions[i][1],
                                   nodeArray[i]);
            }
            
            // Title
            g2.setFont(FONT_BODY);
            g2.setColor(TEXT_MUTED);
            g2.drawString("Federated Network Topology", 10, 20);
            
            // Legend
            drawLegend(g2, w - 160, h - 90);
            
            g2.dispose();
        }
        
        private void drawConnection(Graphics2D g2, int x1, int y1, int x2, int y2,
                                    Color color, boolean isBlocked) {
            if (isBlocked) {
                // Dashed red line for blocked connections
                float[] dash = {8f, 8f};
                g2.setStroke(new BasicStroke(2, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                                           1f, dash, (float)(frameCount % 16)));
                g2.setColor(ALERT_CRITICAL.darker());
            } else {
                // Solid glowing line
                g2.setStroke(new BasicStroke(2));
                g2.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 60));
            }
            
            g2.drawLine(x1, y1, x2, y2);
            g2.setStroke(new BasicStroke(1));
        }
        
        private void drawAggregatorNode(Graphics2D g2, int cx, int cy) {
            int size = 50;
            
            // Outer glow
            float pulse = (float)(0.5 + 0.5 * Math.sin(frameCount * 0.05));
            g2.setColor(new Color(AGGREGATOR_COLOR.getRed(), AGGREGATOR_COLOR.getGreen(),
                       AGGREGATOR_COLOR.getBlue(), (int)(30 + 20 * pulse)));
            g2.fillOval(cx - size - 10, cy - size - 10, (size + 10) * 2, (size + 10) * 2);
            
            // Main circle
            GradientPaint grad = new GradientPaint(
                cx - size, cy - size, AGGREGATOR_COLOR,
                cx + size, cy + size, AGGREGATOR_COLOR.darker()
            );
            g2.setPaint(grad);
            g2.fillOval(cx - size, cy - size, size * 2, size * 2);
            
            // Border
            g2.setColor(AGGREGATOR_COLOR.brighter());
            g2.setStroke(new BasicStroke(2));
            g2.drawOval(cx - size, cy - size, size * 2, size * 2);
            
            // Label
            g2.setFont(FONT_HEADING);
            g2.setColor(TEXT_PRIMARY);
            String label = "AGG";
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(label, cx - fm.stringWidth(label) / 2, cy + 5);
            
            g2.setFont(FONT_SMALL);
            g2.setColor(TEXT_SECONDARY);
            String subLabel = "Meta-Model";
            fm = g2.getFontMetrics();
            g2.drawString(subLabel, cx - fm.stringWidth(subLabel) / 2, cy + 20);
        }
        
        private void drawInstitutionNode(Graphics2D g2, int x, int y, NodeInfo node) {
            int size = 40;
            
            // Outer glow (red if blocked)
            Color glowColor = node.isBlocked ? ALERT_CRITICAL : node.color;
            g2.setColor(new Color(glowColor.getRed(), glowColor.getGreen(),
                       glowColor.getBlue(), 40));
            g2.fillOval(x - size - 8, y - size - 8, (size + 8) * 2, (size + 8) * 2);
            
            // Main circle
            Color fillColor = node.isBlocked ?
                new Color(80, 20, 20) : BG_TERTIARY;
            g2.setColor(fillColor);
            g2.fillOval(x - size, y - size, size * 2, size * 2);
            
            // Border (thickness based on trust)
            float borderWidth = (float)(1 + node.trustScore * 2);
            g2.setStroke(new BasicStroke(borderWidth));
            g2.setColor(node.isBlocked ? ALERT_CRITICAL : node.color);
            g2.drawOval(x - size, y - size, size * 2, size * 2);
            
            // Label
            g2.setFont(FONT_HEADING);
            g2.setColor(node.isBlocked ? ALERT_CRITICAL : TEXT_PRIMARY);
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(node.shortName, x - fm.stringWidth(node.shortName) / 2, y + 5);
            
            // Trust percentage below
            g2.setFont(FONT_SMALL);
            String trustStr = node.isBlocked ? "BLOCKED" :
                String.format("%.0f%%", node.trustScore * 100);
            g2.setColor(node.isBlocked ? ALERT_CRITICAL :
                       (node.trustScore > 0.5 ? RISK_LOW : RISK_HIGH));
            fm = g2.getFontMetrics();
            g2.drawString(trustStr, x - fm.stringWidth(trustStr) / 2, y + size + 16);
            
            // Blocked X overlay
            if (node.isBlocked) {
                g2.setStroke(new BasicStroke(4));
                g2.setColor(ALERT_CRITICAL);
                g2.drawLine(x - 20, y - 20, x + 20, y + 20);
                g2.drawLine(x + 20, y - 20, x - 20, y + 20);
            }
            
            g2.setStroke(new BasicStroke(1));
        }
        
        private void drawLegend(Graphics2D g2, int x, int y) {
            g2.setFont(FONT_SMALL);
            g2.setColor(TEXT_MUTED);
            g2.drawString("Legend:", x, y);
            
            int ly = y + 15;
            Color[] colors = {RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL};
            String[] labels = {"Low Risk", "Medium", "High", "Critical"};
            
            for (int i = 0; i < colors.length; i++) {
                g2.setColor(colors[i]);
                g2.fillRect(x, ly + i * 15 - 8, 10, 10);
                g2.setColor(TEXT_SECONDARY);
                g2.drawString(labels[i], x + 15, ly + i * 15);
            }
        }
        
        /** Add an animated particle traveling between two points */
        void addParticle(int sx, int sy, int ex, int ey, Color color) {
            particles.add(new AnimatedParticle(sx, sy, ex, ey, color, 60));
        }
        
        private void updateParticles() {
            Iterator<AnimatedParticle> it = particles.iterator();
            while (it.hasNext()) {
                AnimatedParticle p = it.next();
                p.update();
                if (p.isDone()) {
                    it.remove();
                }
            }
        }
    }
    
    /** Animated particle for network traffic visualization */
    static class AnimatedParticle {
        double x, y;
        double sx, sy, ex, ey;
        Color color;
        double progress = 0;
        double speed;
        double alpha = 1.0;
        
        AnimatedParticle(int sx, int sy, int ex, int ey, Color color, int frames) {
            this.sx = sx; this.sy = sy;
            this.ex = ex; this.ey = ey;
            this.x = sx; this.y = sy;
            this.color = color;
            this.speed = 1.0 / frames;
        }
        
        void update() {
            progress += speed;
            x = sx + (ex - sx) * progress;
            y = sy + (ey - sy) * progress;
            alpha = progress < 0.8 ? 1.0 : (1.0 - progress) / 0.2;
        }
        
        boolean isDone() { return progress >= 1.0; }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // SIMULATION ENGINE
    // ═══════════════════════════════════════════════════════════════════
    
    /** Start the federated scoring simulation */
    private void startSimulation() {
        if (isSimulating) return;
        isSimulating = true;
        simulationStep = 0;
        
        progressBar.setVisible(true);
        statusLabel.setText("Simulation running...");
        
        Random rng = new Random(42);
        
        simulationTimer = new Timer();
        simulationTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (simulationStep >= TOTAL_SIMULATION_STEPS) {
                    simulationTimer.cancel();
                    SwingUtilities.invokeLater(() -> {
                        isSimulating = false;
                        progressBar.setVisible(false);
                        statusLabel.setText("Simulation complete — " +
                            riskResults.size() + " customers scored");
                    });
                    return;
                }
                
                simulationStep++;
                int progress = (simulationStep * 100) / TOTAL_SIMULATION_STEPS;
                
                SwingUtilities.invokeLater(() -> {
                    progressBar.setValue(progress);
                    processSimulationStep(simulationStep, rng);
                });
            }
        }, 200, 200);
    }
    
    /** Process one step of the simulation */
    private void processSimulationStep(int step, Random rng) {
        String customerId = String.format("CUST_%05d", step);
        String hashedId = hashId(customerId);
        String shortHash = hashedId.substring(0, 12) + "...";
        
        // Phase 1: Request scores from nodes
        if (step % 3 == 0) {
            addTrafficEvent("AGG", "BANK", "SCORE_REQUEST",
                "Requesting score for " + shortHash, AGGREGATOR_COLOR);
        }
        if (step % 3 == 1) {
            addTrafficEvent("AGG", "LEND", "SCORE_REQUEST",
                "Requesting score for " + shortHash, AGGREGATOR_COLOR);
        }
        if (step % 3 == 2) {
            addTrafficEvent("AGG", "INSR", "SCORE_REQUEST",
                "Requesting score for " + shortHash, AGGREGATOR_COLOR);
        }
        
        // Phase 2: Receive scores
        double bankScore = Math.max(0, Math.min(1, rng.nextGaussian() * 0.3 + 0.4));
        double lendScore = Math.max(0, Math.min(1, rng.nextGaussian() * 0.3 + 0.45));
        double insrScore = Math.max(0, Math.min(1, rng.nextGaussian() * 0.3 + 0.35));
        
        // Apply DP rounding
        bankScore = Math.round(bankScore * 10.0) / 10.0;
        lendScore = Math.round(lendScore * 10.0) / 10.0;
        insrScore = Math.round(insrScore * 10.0) / 10.0;
        
        addTrafficEvent("BANK", "AGG", "SCORE_RESPONSE",
            String.format("Score: %.1f (DP applied, TLS 1.3)", bankScore), BANK_COLOR);
        addTrafficEvent("LEND", "AGG", "SCORE_RESPONSE",
            String.format("Score: %.1f (DP applied, TLS 1.3)", lendScore), LENDING_COLOR);
        addTrafficEvent("INSR", "AGG", "SCORE_RESPONSE",
            String.format("Score: %.1f (DP applied, TLS 1.3)", insrScore), INSURER_COLOR);
        
        // Update node stats
        nodes.get("bank").scoresSent++;
        nodes.get("lending_app").scoresSent++;
        nodes.get("insurer").scoresSent++;
        
        // Aggregate
        double finalScore = bankScore * 0.45 + lendScore * 0.25 + insrScore * 0.30;
        finalScore = Math.round(finalScore * 10000.0) / 10000.0;
        
        String tier;
        if (finalScore < 0.25) tier = "LOW";
        else if (finalScore < 0.50) tier = "MEDIUM";
        else if (finalScore < 0.75) tier = "HIGH";
        else tier = "CRITICAL";
        
        // Add to results table
        RiskResult result = new RiskResult();
        result.hashedId = hashedId;
        result.finalScore = finalScore;
        result.riskTier = tier;
        result.nodeScores.put("bank", bankScore);
        result.nodeScores.put("lending_app", lendScore);
        result.nodeScores.put("insurer", insrScore);
        riskResults.add(result);
        
        riskTableModel.addRow(new Object[]{
            shortHash,
            String.format("%.4f", finalScore),
            tier,
            String.format("%.1f", bankScore),
            String.format("%.1f", lendScore),
            String.format("%.1f", insrScore),
            "1.00"
        });
        
        // Scroll to bottom
        int lastRow = riskTable.getRowCount() - 1;
        riskTable.scrollRectToVisible(riskTable.getCellRect(lastRow, 0, true));
        
        // Add network particles
        int w = networkPanel.getWidth();
        int h = networkPanel.getHeight();
        int cx = w / 2, cy = h / 2;
        int r = Math.min(w, h) / 3;
        
        networkPanel.addParticle(cx - r, cy - r/2, cx, cy, BANK_COLOR);
        networkPanel.addParticle(cx + r, cy - r/2, cx, cy, LENDING_COLOR);
        networkPanel.addParticle(cx, cy + r, cx, cy, INSURER_COLOR);
        
        // Refresh node cards
        refreshNodeCards();
    }
    
    /** Simulate a poisoning attack */
    private void simulateAttack() {
        NodeInfo insurer = nodes.get("insurer");
        
        addTrafficEvent("INSR", "AGG", "ALERT",
            "⚠ ANOMALOUS SUBMISSION DETECTED: Scores inverted!", ALERT_CRITICAL);
        
        // Flash alert
        flashAlert();
        
        String alertMsg = String.format(
            "[%s] 🚨 POISONING ATTACK DETECTED\n" +
            "  Node: Insurer (IRDAI)\n" +
            "  Type: Score Inversion Attack\n" +
            "  Correlation with reference: -0.847 (threshold: -0.3)\n" +
            "  Action: Submission BLOCKED\n" +
            "  Trust: %.0f%% → %.0f%%\n\n",
            new java.text.SimpleDateFormat("HH:mm:ss").format(new java.util.Date()),
            insurer.trustScore * 100,
            insurer.trustScore * 50
        );
        
        alertTextArea.append(alertMsg);
        alertTextArea.setCaretPosition(alertTextArea.getDocument().getLength());
        
        // Update node status
        insurer.trustScore *= 0.5;
        if (insurer.trustScore < 0.25) {
            insurer.isBlocked = true;
            insurer.status = "BLOCKED";
            
            alertTextArea.append(
                "[" + new java.text.SimpleDateFormat("HH:mm:ss")
                    .format(new java.util.Date()) +
                "] 🚫 NODE BLOCKED: Insurer trust below threshold (25%)\n" +
                "  All future submissions from this node will be rejected.\n\n"
            );
        }
        
        insurer.scoresRejected += 100;
        
        addTrafficEvent("AGG", "INSR", "BLOCKED",
            "NODE BLOCKED — Trust below threshold", ALERT_BLOCKED);
        
        refreshNodeCards();
        statusLabel.setText("⚡ Attack intercepted — Insurer node blocked!");
    }
    
    /** Flash the alert bar red */
    private void flashAlert() {
        Timer flashTimer = new Timer();
        final int[] flashCount = {0};
        
        flashTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                SwingUtilities.invokeLater(() -> {
                    if (flashCount[0] % 2 == 0) {
                        alertFlashPanel.setBackground(ALERT_CRITICAL);
                        alertFlashPanel.setPreferredSize(new Dimension(0, 4));
                    } else {
                        alertFlashPanel.setBackground(BG_SECONDARY);
                        alertFlashPanel.setPreferredSize(new Dimension(0, 3));
                    }
                    alertFlashPanel.getParent().revalidate();
                });
                
                flashCount[0]++;
                if (flashCount[0] > 10) {
                    flashTimer.cancel();
                    SwingUtilities.invokeLater(() -> {
                        alertFlashPanel.setBackground(BG_SECONDARY);
                        alertFlashPanel.setPreferredSize(new Dimension(0, 3));
                        alertFlashPanel.getParent().revalidate();
                    });
                }
            }
        }, 0, 150);
    }
    
    /** Add a traffic event to the log */
    private void addTrafficEvent(String src, String dst, String type, String msg, Color color) {
        TrafficEvent event = new TrafficEvent(src, dst, type, msg, color);
        trafficLog.add(event);
        
        String timestamp = new java.text.SimpleDateFormat("HH:mm:ss.SSS")
            .format(new java.util.Date());
        
        String prefix;
        switch (type) {
            case "SCORE_REQUEST":  prefix = "→ REQ "; break;
            case "SCORE_RESPONSE": prefix = "← RES "; break;
            case "ALERT":          prefix = "⚠ ALT "; break;
            case "BLOCKED":        prefix = "🚫 BLK"; break;
            default:               prefix = "  --- "; break;
        }
        
        trafficTextArea.append(String.format(
            "[%s] %s [%s→%s] %s\n", timestamp, prefix, src, dst, msg
        ));
        trafficTextArea.setCaretPosition(trafficTextArea.getDocument().getLength());
    }
    
    /** Reset the dashboard to initial state */
    private void resetDashboard() {
        if (simulationTimer != null) simulationTimer.cancel();
        isSimulating = false;
        simulationStep = 0;
        
        trafficLog.clear();
        riskResults.clear();
        alertLog.clear();
        
        trafficTextArea.setText("");
        alertTextArea.setText("");
        riskTableModel.setRowCount(0);
        
        for (NodeInfo node : nodes.values()) {
            node.trustScore = 1.0;
            node.isBlocked = false;
            node.scoresSent = 0;
            node.scoresRejected = 0;
            node.status = "ACTIVE";
        }
        
        progressBar.setValue(0);
        progressBar.setVisible(false);
        statusLabel.setText("Dashboard reset — Ready");
        
        refreshNodeCards();
    }
    
    /** Load results from Python backend output files */
    private void loadResultsFromFile() {
        JFileChooser chooser = new JFileChooser(".");
        chooser.setDialogTitle("Select aggregated_risk_scores.csv");
        chooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("CSV Files", "csv"));
        
        int result = chooser.showOpenDialog(this);
        if (result != JFileChooser.APPROVE_OPTION) return;
        
        File file = chooser.getSelectedFile();
        statusLabel.setText("Loading results from " + file.getName() + "...");
        
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null) return;
            
            String line;
            int count = 0;
            while ((line = reader.readLine()) != null && count < 200) {
                String[] parts = line.split(",");
                if (parts.length >= 7) {
                    String hashId = parts[0].length() > 15 ?
                        parts[0].substring(0, 12) + "..." : parts[0];
                    
                    riskTableModel.addRow(new Object[]{
                        hashId,
                        parts[1],   // final_risk_score
                        parts[2],   // risk_tier
                        parts.length > 5 ? parts[5] : "—",  // bank score
                        parts.length > 6 ? parts[6] : "—",  // lending score
                        parts.length > 7 ? parts[7] : "—",  // insurer score
                        parts[3],   // confidence
                    });
                    count++;
                }
            }
            
            statusLabel.setText("Loaded " + count + " risk scores from " + file.getName());
            
            // Try to load alerts file from same directory
            File alertFile = new File(file.getParent(), "security_alerts.json");
            if (alertFile.exists()) {
                loadAlerts(alertFile);
            }
            
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this,
                "Error loading file: " + e.getMessage(),
                "Load Error", JOptionPane.ERROR_MESSAGE);
        }
    }
    
    /** Load security alerts from JSON file */
    private void loadAlerts(File alertFile) {
        try {
            String content = new String(Files.readAllBytes(alertFile.toPath()));
            // Simple JSON parsing (no external library needed)
            // Just display raw content since we don't have a JSON parser
            alertTextArea.append("=== Loaded Alerts from " + alertFile.getName() + " ===\n\n");
            
            // Basic extraction of alert messages
            String[] lines = content.split("\n");
            for (String line : lines) {
                String trimmed = line.trim();
                if (trimmed.contains("\"message\"")) {
                    String msg = trimmed.replace("\"message\":", "")
                                       .replace("\"", "").replace(",", "").trim();
                    alertTextArea.append("⚠ " + msg + "\n");
                }
                if (trimmed.contains("\"severity\"")) {
                    String sev = trimmed.replace("\"severity\":", "")
                                       .replace("\"", "").replace(",", "").trim();
                    if (sev.equals("CRITICAL") || sev.equals("BLOCKED")) {
                        flashAlert();
                    }
                }
            }
        } catch (IOException e) {
            // Silently skip alert loading
        }
    }
    
    /** Refresh the node status cards */
    private void refreshNodeCards() {
        // Rebuild the node cards panel
        // Since we can't easily update individual cards, we repaint
        repaint();
    }
    
    /** Simple SHA-256 hash for simulation */
    private String hashId(String input) {
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest((input + ":JabsonX_FedRisk_2026_DPDP_Compliant").getBytes());
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            return input;
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // MAIN ENTRY POINT
    // ═══════════════════════════════════════════════════════════════════
    
    public static void main(String[] args) {
        // Set system properties for better rendering
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");
        
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                // Use default look and feel
            }
            
            new FederatedDashboard();
        });
    }
}
