import Nutmeg.Gui 0.1
import QtQuick 2.4

Gui {
    Column {
        width: parent.width

        Slider {
            handle: "planez"
            text: "Plane Z"
            decimals: 3
            minimumValue: 0.0
            maximumValue: 1.0
            stepSize: 0.001
        }

        Slider {
            handle: "eps"
            text: "Tolerance"
            minimumValue: 0.0
            maximumValue: 0.1
            stepSize: 0.001
        }
    }
}
