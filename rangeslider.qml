import QtQuick 2.2
import QtQuick.Controls 1.2

Item {
    id: page
    width: 300
    height: 400

    Column {
        spacing: 40
        width: parent.width

        Label {
            width: parent.width
            wrapMode: Label.Wrap
            text: "This slider is used to select a range specified by two values, by sliding each handle along a track. It is also possible to drag the range to change both the lower and upper values while keeping the range size the same."
        }
        Label {
            width: parent.width
            wrapMode: Label.Wrap
            text: "From %1 to %2 (%3)".arg(rangeSlider.lower).arg(rangeSlider.upper).arg(rangeSlider.range)
        }

        Item {
            id: rangeSlider

            width: parent.width
            property real handleSize: 20
            property real handleRadius: handleSize / 2
            property real minimumRangeHandleSize: 8
            property real trackThickness: 5
            property real trackRadius: trackThickness / 2
            property real trackBorderWidth: 1
            property color lowerHandleColor: "green"
            property color upperHandleColor: "red"
            property color rangeHandleColor: "grey"
            property color trackColor: "white"
            property color trackBorderColor: "grey"

            property real from: 0
            property real to: 100
            property real minimumRange: 1
            property bool roundValues: true

            property real lower:
            {
                var result = lowerHandle.x / (width - (2 * handleSize + minimumRangeHandleSize));
                result = from + result * (to - (from + minimumRange));
                result = roundValues ? Math.round(result) | 0 : result;
                return result;
            }
            property real upper:
            {
                var result = (upperHandle.x - (handleSize + minimumRangeHandleSize)) / (width - (2 * handleSize + minimumRangeHandleSize));
                result = from + minimumRange + result * (to - (from + minimumRange));
                result = roundValues ? Math.round(result) : result;
                return result;
            }
            property real range: upper - lower
            property var activeHandle: upperHandle

            height: handleSize

            Rectangle {
                width: parent.width - parent.handleSize
                height: parent.trackThickness
                radius: parent.trackRadius
                anchors.centerIn: parent
                color: parent.trackColor
                border.width: parent.trackBorderWidth;
                border.color: parent.trackBorderColor;
            }

            Item {
                id: rangeHandle
                x: lowerHandle.x + lowerHandle.width
                width: parent.minimumRangeHandleSize
                height: parent.handleSize
                property real value: parent.upper

                Rectangle {
                    anchors.centerIn: parent
                    height: parent.parent.trackThickness - 2 * parent.parent.trackBorderWidth
                    width: parent.width + parent.parent.handleSize
                    color: parent.parent.rangeHandleColor
                }

                MouseArea {
                    anchors.fill: parent

                    drag.target: parent
                    drag.axis: Drag.XAxis
                    drag.minimumX: lowerHandle.width
                    drag.maximumX: parent.parent.width - (parent.width + upperHandle.width)

                    onPressed: parent.parent.activeHandle = rangeHandle
                    onPositionChanged:
                    {
                        lowerHandle.x = parent.x - lowerHandle.width
                        upperHandle.x = parent.x + parent.width
                    }
                }
            }

            Rectangle {
                id: lowerHandle
                x: 0
                width: parent.handleSize
                height: parent.handleSize
                radius: parent.handleRadius
                color: parent.lowerHandleColor
                property real value: parent.lower

                MouseArea {
                    anchors.fill: parent

                    drag.target: parent
                    drag.axis: Drag.XAxis
                    drag.minimumX: 0
                    drag.maximumX: parent.parent.width - (parent.width + upperHandle.width + parent.parent.minimumRangeHandleSize)

                    onPressed: parent.parent.activeHandle = lowerHandle
                    onPositionChanged:
                    {
                        if(upperHandle.x - (lowerHandle.x + lowerHandle.width) < parent.parent.minimumRangeHandleSize)
                        {
                            upperHandle.x = lowerHandle.x + lowerHandle.width + parent.parent.minimumRangeHandleSize;
                        }
                        rangeHandle.width = upperHandle.x - (lowerHandle.x + lowerHandle.width);
                    }
                }
            }

            Rectangle {
                id: upperHandle
                x: parent.minimumRangeHandleSize + parent.handleSize
                width: parent.handleSize
                height: parent.handleSize
                radius: parent.handleRadius
                color: parent.upperHandleColor
                property real value: parent.upper

                MouseArea {
                    anchors.fill: parent

                    drag.target: parent
                    drag.axis: Drag.XAxis
                    drag.minimumX: lowerHandle.width + parent.parent.minimumRangeHandleSize
                    drag.maximumX: parent.parent.width - parent.width

                    onPressed: parent.parent.activeHandle = upperHandle
                    onPositionChanged:
                    {
                        if(upperHandle.x - (lowerHandle.x + lowerHandle.width) < parent.parent.minimumRangeHandleSize)
                        {
                            lowerHandle.x = upperHandle.x - (lowerHandle.width + parent.parent.minimumRangeHandleSize);
                        }
                        rangeHandle.width = upperHandle.x - (lowerHandle.x + lowerHandle.width)
                    }
                }
            }
        }

        Rectangle
        {
            id: label
            x: rangeSlider.activeHandle.x + (rangeSlider.activeHandle.width / 2) - 25
            y: rangeSlider.y + rangeSlider.height + 5
            border.width: 1
            border.color: "grey"
            TextField
            {
                width: 50
                text: rangeSlider.activeHandle.value
                horizontalAlignment: TextInput.AlignHCenter
            }
        }
    }
}
