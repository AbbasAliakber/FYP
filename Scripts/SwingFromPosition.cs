using UnityEngine;

public class SwingFromPosition : MonoBehaviour
{
    public float amplitude = 2f;
    public float frequency = 1f;
    public Vector3 axis = Vector3.forward;
    void Update()
    {
        float angle = amplitude * Mathf.Sin(frequency * Time.time);
        transform.rotation = Quaternion.AngleAxis(angle, axis);
    }
}
