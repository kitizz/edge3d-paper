import Nutmeg 0.1

Figure {
    Column {
        anchors.fill: parent
        // rows: 2
        // layoutDirection: "Vertical"
        // rowWeights: [0.5, 0.5]

        Axis {
            handle: "xaxis"
            shareX: "axis"

            LinePlot {
                handle: "P1"
                line { color: "#2222FF"; width: 3; style: "-" }
            }

            LinePlot {
                handle: "P2"
                line { color: "#FF2222"; width: 3; style: "--" }
            }

            LinePlot {
                handle: "valid"
                line { color: "#22FF22"; width: 2; style: "-" }
            }
        }

        Axis {
            handle: "yaxis"
            shareX: "axis"

            LinePlot {
                handle: "P1"
                line { color: "#2222FF"; width: 3; style: "-" }
            }

            LinePlot {
                handle: "P2"
                line { color: "#FF2222"; width: 3; style: "--" }
            }

            LinePlot {
                handle: "valid"
                line { color: "#22FF22"; width: 2; style: "-" }
            }
        }

        Axis {
            handle: "zaxis"
            shareX: "axis"

            LinePlot {
                handle: "P1"
                line { color: "#2222FF"; width: 3; style: "-" }
            }

            LinePlot {
                handle: "P2"
                line { color: "#FF2222"; width: 3; style: "--" }
            }

            LinePlot {
                handle: "valid"
                line { color: "#22FF22"; width: 2; style: "-" }
            }
        }
    }
}
