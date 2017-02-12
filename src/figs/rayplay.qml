import Nutmeg 0.1

Figure {
    Axis {
        handle: "ax"
        aspectRatio: 1

        LineSegmentPlot {
            handle: "rays"
            line { width: 2; color: "#664444FF" }
        }

        LineSegmentPlot {
            handle: "rays2"
            line { width: 2; color: "#1133DD33" }
        }

        LinePlot {
            handle: "circle"
            line { width: 4; color: "#99FF4444"; style: "--" }
        }

        LinePlot {
            handle: "P0"
            line { width: 4; color: "#994444FF"; style: "." }
        }

        LinePlot {
            handle: "P1"
            line { width: 2; color: "#FF4444"; style: "." }
        }
    }
}
