import Nutmeg 0.1

Figure {
    Axis {
        handle: "ax"
        // aspectRatio: 1

        LinePlot {
            handle: "l0"
            line { width: 2; color: "#4444FF"; style: "-" }
        }

        LinePlot {
            handle: "l1"
            line { width: 2; color: "#FF4444"; style: "--" }
        }
    }
}
