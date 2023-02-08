using UnityEngine;

public class SwingObject : MonoBehaviour
{
    public float amplitude = 20f;
    public float frequency = 1f;
    private Vector3 startPos;

    void Start()
    {
        startPos = transform.position;
    }

    void Update()
    {
        float x = amplitude * Mathf.Sin(frequency * Time.time);
        transform.position = startPos + new Vector3(x, 0, 0);
    }
}
